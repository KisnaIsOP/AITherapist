from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import random

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///nirya.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    interactions = db.relationship('Interaction', backref='user', lazy=True)
    emotions = db.relationship('EmotionRecord', backref='user', lazy=True)

class Interaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    message = db.Column(db.Text)
    response = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    feedback_score = db.Column(db.Integer, nullable=True)
    feedback_text = db.Column(db.Text, nullable=True)

class EmotionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    emotions = db.Column(db.String(200))  # Stored as JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class GratitudeEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Analytics Helper Functions
def get_or_create_user():
    if 'user_id' not in session:
        user = User(session_id=os.urandom(24).hex())
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
    else:
        user = User.query.get(session['user_id'])
        user.last_active = datetime.utcnow()
        db.session.commit()
    return user

def log_interaction(user, message, response, emotions=None):
    # Log the interaction
    interaction = Interaction(
        user_id=user.id,
        message=message,
        response=response
    )
    db.session.add(interaction)
    
    # Log emotions if present
    if emotions:
        emotion_record = EmotionRecord(
            user_id=user.id,
            emotions=json.dumps(emotions)
        )
        db.session.add(emotion_record)
    
    db.session.commit()
    return interaction

def get_user_analytics():
    total_users = User.query.count()
    active_users = User.query.filter(
        User.last_active >= datetime.utcnow() - timedelta(days=7)
    ).count()
    
    # Get common emotions
    emotions = EmotionRecord.query.all()
    emotion_counts = {}
    for record in emotions:
        for emotion in json.loads(record.emotions):
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # Get average feedback score
    avg_score = db.session.query(db.func.avg(Interaction.feedback_score))\
        .filter(Interaction.feedback_score.isnot(None))\
        .scalar()
    
    return {
        'total_users': total_users,
        'active_users': active_users,
        'common_emotions': emotion_counts,
        'average_feedback': round(avg_score, 2) if avg_score else 0
    }

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
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    # Get or create user
    user = get_or_create_user()
    
    # Get conversation history
    conversation_history = session.get('conversation_history', [])
    history_text = "\n".join([f"{'User' if i%2==0 else 'Nirya'}: {msg}" 
                             for i, msg in enumerate(conversation_history)])

    # Get AI response
    ai_response = get_ai_response(user_message, history_text)

    # Log interaction and emotions
    current_emotions = analyze_emotion(user_message)
    interaction = log_interaction(user, user_message, ai_response, current_emotions)

    # Update conversation history
    conversation_history.extend([user_message, ai_response])
    session['conversation_history'] = conversation_history[-10:]

    return jsonify({
        'response': ai_response,
        'timestamp': datetime.now().strftime("%H:%M"),
        'interaction_id': interaction.id
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    if 'user_id' in session:
        interaction_id = feedback_data.get('interaction_id')
        if interaction_id:
            interaction = Interaction.query.get(interaction_id)
            if interaction:
                interaction.feedback_score = feedback_data.get('score')
                interaction.feedback_text = feedback_data.get('feedback')
                db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/journal', methods=['POST'])
def save_journal():
    if 'user_id' in session:
        content = request.json.get('content')
        if content:
            entry = JournalEntry(
                user_id=session['user_id'],
                content=content
            )
            db.session.add(entry)
            db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/gratitude', methods=['POST'])
def save_gratitude():
    if 'user_id' in session:
        content = request.json.get('content')
        if content:
            entry = GratitudeEntry(
                user_id=session['user_id'],
                content=content
            )
            db.session.add(entry)
            db.session.commit()
    return jsonify({'status': 'success'})

# Admin dashboard route (protected)
@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    analytics = get_user_analytics()
    return render_template('admin/analytics.html', analytics=analytics)

if __name__ == '__main__':
    app.run(debug=True)
