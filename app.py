from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from functools import wraps
import hashlib
import random
import json
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
from collections import defaultdict, Counter

load_dotenv()

# Initialize Flask and Socket.IO with proper configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    manage_session=True,
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8
)

# Simplified User model for temporary tracking
class User:
    def __init__(self, session_id=None):
        self.id = os.urandom(24).hex()
        self.session_id = session_id or self.id
        self.created_at = datetime.utcnow()
        self.last_active = self.created_at
        self.interactions = []
        self.emotions = []

# Modify get_or_create_user to use in-memory tracking
def get_or_create_user():
    if 'user_id' not in session:
        user = User()
        session['user_id'] = user.id
    else:
        user = User(session_id=session['user_id'])
    
    return user

# Modify log_interaction to store in memory
def log_interaction(user, message, response, emotions=None):
    interaction = {
        'message': message,
        'response': response,
        'timestamp': datetime.utcnow()
    }
    user.interactions.append(interaction)
    
    if emotions:
        user.emotions.append({
            'emotions': emotions,
            'timestamp': datetime.utcnow()
        })
    
    return interaction

# In-memory analytics storage
class Analytics:
    def __init__(self):
        self.total_sessions = 0
        self.active_sessions = set()
        self.emotions = []
        self.feedback_scores = []
        self.conversation_lengths = []
        self.hourly_usage = defaultdict(int)
        self.daily_usage = defaultdict(int)
        self.response_times = []
        self.user_locations = Counter()
        self.common_topics = Counter()
        
    def add_session(self, session_id):
        self.total_sessions += 1
        self.active_sessions.add(session_id)
        self.update_usage_metrics()
        self.broadcast_updates()
    
    def remove_session(self, session_id):
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
        self.broadcast_updates()
    
    def add_emotion(self, emotion):
        self.emotions.append(emotion)
        self.broadcast_updates()
    
    def add_feedback(self, score):
        self.feedback_scores.append(score)
        self.broadcast_updates()
    
    def add_conversation_length(self, length):
        self.conversation_lengths.append(length)
        self.broadcast_updates()
    
    def add_response_time(self, time):
        self.response_times.append(time)
        self.broadcast_updates()
    
    def add_topic(self, topic):
        self.common_topics[topic] += 1
        self.broadcast_updates()
    
    def update_usage_metrics(self):
        current_hour = datetime.now().strftime('%H:00')
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.hourly_usage[current_hour] += 1
        self.daily_usage[current_date] += 1
    
    def get_analytics_data(self):
        return {
            'total_sessions': self.total_sessions,
            'active_sessions': len(self.active_sessions),
            'common_emotions': dict(Counter(self.emotions).most_common(5)),
            'average_feedback': sum(self.feedback_scores) / len(self.feedback_scores) if self.feedback_scores else 0,
            'avg_conversation_length': sum(self.conversation_lengths) / len(self.conversation_lengths) if self.conversation_lengths else 0,
            'hourly_usage': dict(self.hourly_usage),
            'daily_usage': dict(self.daily_usage),
            'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            'top_locations': dict(self.user_locations.most_common(5)),
            'common_topics': dict(self.common_topics.most_common(10))
        }
    
    def broadcast_updates(self):
        socketio.emit('analytics_update', self.get_analytics_data(), namespace='/admin')

# Initialize analytics
analytics = Analytics()

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini AI model
model = genai.GenerativeModel('gemini-pro')

# Enhanced therapeutic conversation context with advanced engagement strategies
THERAPEUTIC_CONTEXT = """
You are an empathetic and engaging AI therapist. Follow these guidelines for meaningful conversations:

1. Active Listening & Paraphrasing:
- Mirror their words: "I hear you saying that..."
- Validate feelings: "It makes sense that you feel [emotion] because..."
- Check understanding: "Let me make sure I understand correctly..."

2. Emotional Reflection:
- Acknowledge emotions explicitly: "It sounds like you're feeling [emotion]"
- Show deep understanding: "That must be really [challenging/frustrating/difficult]"
- Validate experiences: "It's completely natural to feel this way"

3. Personalized Engagement:
- Reference previous details: "Earlier you mentioned [detail]..."
- Use their language style: Match formal/casual tone
- Remember key points: "Last time we talked about [topic]..."

4. Storytelling Encouragement:
- Ask for specific examples: "Can you tell me about a time when..."
- Explore recent experiences: "What was the last situation that made you feel this way?"
- Encourage details: "What happened next?"

5. Gentle Probing Techniques:
- Use follow-up questions: "What do you think led to that?"
- Explore patterns: "Have you noticed when these feelings are strongest?"
- Investigate coping: "How do you usually handle such situations?"

6. Collaborative Problem-Solving:
- Brainstorm together: "Let's explore some possibilities..."
- Ask for their input: "What solutions have you considered?"
- Build on their ideas: "That's an interesting approach. How might we develop that?"

7. Trust Building:
- Start light: Begin with easier topics
- Progress gradually: Move to deeper issues as trust builds
- Acknowledge sharing: "Thank you for trusting me with this"

8. Support System Exploration:
- Ask about resources: "Who do you turn to for support?"
- Explore relationships: "How do others in your life view this?"
- Discuss professional help when appropriate

9. Interactive Engagement:
- Suggest reflection exercises: "Would you like to try a quick mindfulness exercise?"
- Set micro-goals: "What's one small step you could take today?"
- Encourage journaling: "Consider writing down your thoughts about..."

10. Conversation Flow:
- Use open-ended questions
- Build on their responses
- Maintain natural dialogue
- Show genuine curiosity

Important Guidelines:
- Always maintain empathy and support
- Never provide medical advice
- Keep responses focused and relevant
- Encourage professional help when needed
- Celebrate progress and insights
- Handle crisis situations appropriately
"""

# Expanded engagement questions for different scenarios
ENGAGEMENT_QUESTIONS = {
    'initial': [
        "I'm here to listen and support you. What brings you here today?",
        "Before we begin, how are you feeling right now? Take your time to share.",
        "I'd love to understand what's been going on for you lately. What would you like to explore?",
        "Thank you for reaching out. What's been on your mind that you'd like to discuss?"
    ],
    'follow_up': [
        "How have things been since we last talked?",
        "What's been most on your mind since our last conversation?",
        "I remember we discussed [topic]. How have things developed since then?"
    ],
    'deepening': [
        "Could you tell me more about that experience?",
        "What do you think that situation taught you about yourself?",
        "How did that make you feel in the moment?"
    ],
    'support': [
        "That sounds really challenging. What kind of support would be most helpful right now?",
        "You're showing a lot of strength by sharing this. What would help you feel more supported?",
        "I'm here with you. What would you like to focus on first?"
    ]
}

# Add tracking for engagement metrics and emotional patterns
class ConversationMetrics:
    def __init__(self):
        self.conversation_length = 0
        self.emotional_patterns = {}
        self.feedback_scores = []
        self.recurring_themes = set()
        self.session_duration = 0
        self.helpful_responses = 0

    def to_dict(self):
        return {
            'conversation_length': self.conversation_length,
            'emotional_patterns': self.emotional_patterns,
            'feedback_scores': self.feedback_scores,
            'recurring_themes': list(self.recurring_themes),
            'session_duration': self.session_duration,
            'helpful_responses': self.helpful_responses
        }

class UserSession:
    def __init__(self):
        self.metrics = ConversationMetrics()
        self.current_emotions = []
        self.themes_discussed = set()
        self.last_topics = []
        self.gratitude_entries = []
        self.journal_entries = []

    def to_dict(self):
        return {
            'metrics': self.metrics.to_dict(),
            'current_emotions': self.current_emotions,
            'themes_discussed': list(self.themes_discussed),
            'last_topics': self.last_topics,
            'gratitude_entries': self.gratitude_entries,
            'journal_entries': self.journal_entries
        }

# Micro-challenges and reflection prompts
MICRO_CHALLENGES = {
    'gratitude': [
        "What are 3 things, big or small, that you're grateful for today?",
        "Can you think of one person who's made a positive impact on your life recently?",
        "What's one small win you've had today that's worth celebrating?"
    ],
    'reflection': [
        "Take a moment to write down what's overwhelming you right now.",
        "What's one small step you could take today toward your goal?",
        "If you could change one thing about your current situation, what would it be?"
    ],
    'mindfulness': [
        "Let's try a quick breathing exercise together.",
        "Take 30 seconds to notice three things you can see, hear, and feel.",
        "What's one thing you can do right now to feel more grounded?"
    ]
}

# Test personas for response calibration
TEST_PERSONAS = {
    'stressed_student': {
        'context': "Struggling with exam pressure and time management",
        'common_themes': ['academic stress', 'time management', 'future anxiety'],
        'sample_queries': [
            "I have three exams next week and I'm freaking out",
            "I can't focus on studying anymore",
            "What if I fail my classes?"
        ]
    },
    'lonely_adult': {
        'context': "Dealing with isolation and low self-confidence",
        'common_themes': ['loneliness', 'self-doubt', 'social anxiety'],
        'sample_queries': [
            "I feel like I have no real friends",
            "It's hard to connect with people",
            "I'm always afraid of saying the wrong thing"
        ]
    },
    'goal_seeker': {
        'context': "Feeling stuck in personal/professional growth",
        'common_themes': ['career uncertainty', 'motivation', 'self-improvement'],
        'sample_queries': [
            "I don't know what I want to do with my life",
            "I feel stuck in my career",
            "How do I stay motivated?"
        ]
    }
}

def analyze_emotion(text):
    """Analyze emotional content of user messages"""
    emotions = {
        'anxiety': ['worried', 'anxious', 'stressed', 'nervous', 'panic'],
        'sadness': ['sad', 'down', 'depressed', 'lonely', 'hopeless'],
        'frustration': ['frustrated', 'angry', 'annoyed', 'upset', 'mad'],
        'hope': ['hopeful', 'excited', 'looking forward', 'better', 'improving'],
        'confusion': ['confused', 'unsure', 'lost', 'don\'t know', 'uncertain']
    }
    
    detected_emotions = []
    text_lower = text.lower()
    
    for emotion, keywords in emotions.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_emotions.append(emotion)
    
    return detected_emotions

def suggest_micro_challenge(emotions, themes):
    """Suggest appropriate micro-challenge based on emotional state and conversation themes"""
    if 'anxiety' in emotions:
        return random.choice(MICRO_CHALLENGES['mindfulness'])
    elif 'sadness' in emotions:
        return random.choice(MICRO_CHALLENGES['gratitude'])
    else:
        return random.choice(MICRO_CHALLENGES['reflection'])

def get_ai_response(user_input, conversation_history):
    try:
        # Initialize or get user session
        if 'user_session' not in session:
            session['user_session'] = UserSession().to_dict()

        # Create the prompt with Nirya's persona
        prompt = f"""You are Nirya, an empathetic AI mental health companion. Your name comes from Sanskrit, meaning wisdom and guidance. 
        You provide supportive, understanding responses while maintaining appropriate boundaries. You're warm, gentle, and focused on emotional well-being.
        
        Previous conversation:
        {conversation_history}
        
        User's message: {user_input}
        
        Respond as Nirya, keeping in mind:
        1. Be empathetic and validate emotions
        2. Use a warm, supportive tone
        3. Ask thoughtful follow-up questions
        4. Suggest gentle reflections when appropriate
        5. Never give medical advice
        6. Maintain professional boundaries
        
        Nirya's response:"""

        response = model.generate_content(prompt)
        
        # Update session metrics
        if any(phrase in response.text.lower() for phrase in ['thank you', 'helps', 'helpful']):
            user_session_data = session['user_session']
            user_session = UserSession()
            user_session.metrics = user_session_data['metrics']
            user_session.metrics.helpful_responses += 1
            session['user_session'] = user_session.to_dict()
        return response.text

    except Exception as e:
        return "I want to make sure I understand what you're sharing. Could you tell me more about that?"

# Admin authentication
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Hash the password for comparison
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Get admin credentials from environment variables
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_password = os.getenv('ADMIN_PASSWORD_HASH')  # Store the hash in .env
        
        if username == admin_username and hashed_password == admin_password:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_analytics'))
        else:
            return render_template('admin/login.html', error="Invalid credentials")
    
    return render_template('admin/login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/')
def home():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
        user_session = UserSession()
        session['user_session'] = user_session.to_dict()
        return render_template('index.html', initial_question=random.choice(ENGAGEMENT_QUESTIONS['initial']))
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = session.get('session_id')

        # Track conversation start time
        if 'conversation_start' not in session:
            session['conversation_start'] = datetime.now().isoformat()
            session['message_count'] = 0

        # Increment message count
        session['message_count'] = session.get('message_count', 0) + 1

        # Basic emotion detection
        emotions = ['happy', 'sad', 'angry', 'anxious', 'grateful', 'confused', 'hopeful', 'frustrated']
        detected_emotion = next((emotion for emotion in emotions if emotion in user_message.lower()), None)
        if detected_emotion:
            analytics.add_emotion(detected_emotion)

        # Topic detection (simplified)
        topics = ['work', 'family', 'health', 'relationships', 'stress', 'depression', 'anxiety', 
                 'self-improvement', 'goals', 'sleep', 'meditation', 'exercise']
        detected_topics = [topic for topic in topics if topic in user_message.lower()]
        for topic in detected_topics:
            analytics.add_topic(topic)

        # Get AI response
        response = get_ai_response(user_message, session_id)

        # Track conversation length if conversation ends
        if 'goodbye' in user_message.lower() or 'bye' in user_message.lower():
            if 'conversation_start' in session:
                start_time = datetime.fromisoformat(session['conversation_start'])
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
                analytics.add_conversation_length(duration)
                session.pop('conversation_start', None)
                session.pop('message_count', None)

        return jsonify({'response': response})

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        rating = data.get('rating')
        
        if rating is not None:
            analytics.add_feedback(float(rating))
            return jsonify({'status': 'success'})
        
        return jsonify({'error': 'Invalid rating'}), 400

    except Exception as e:
        print(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/journal', methods=['POST'])
def save_journal():
    if 'user_id' in session:
        content = request.json.get('content')
        if content:
            # No database to store journal entries, ignore
            pass
    return jsonify({'status': 'success'})

@app.route('/gratitude', methods=['POST'])
def save_gratitude():
    if 'user_id' in session:
        content = request.json.get('content')
        if content:
            # No database to store gratitude entries, ignore
            pass
    return jsonify({'status': 'success'})

# Admin dashboard route (protected)
@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    initial_data = analytics.get_analytics_data()
    return render_template('admin/analytics.html', initial_data=initial_data)

# WebSocket events
@socketio.on('connect', namespace='/admin')
def handle_admin_connect():
    if not session.get('admin_logged_in'):
        return False  # Reject connection if not logged in
    emit('analytics_update', analytics.get_analytics_data())

@app.before_request
def track_analytics():
    if request.endpoint != 'static':
        session_id = session.get('session_id')
        if session_id:
            analytics.add_session(session_id)

@app.after_request
def after_request(response):
    start_time = getattr(request, 'start_time', datetime.now())
    response_time = (datetime.now() - start_time).total_seconds()
    analytics.add_response_time(response_time)
    return response

if __name__ == '__main__':
    socketio.run(app, debug=True)
