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

load_dotenv()

# Initialize Flask and Socket.IO with proper configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

# Configure Socket.IO for production with optimized settings
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=30,
    ping_interval=15,
    logger=False,
    engineio_logger=False,
    path='/socket.io',
    async_handlers=True,
    max_http_buffer_size=5e6  # 5MB max payload
)

# Store chat history in memory (limit to last 25 chats for better memory usage)
MAX_CHAT_HISTORY = 25
chat_history = []

class User:
    active_users = {}

    def __init__(self, session_id=None):
        self.id = os.urandom(24).hex()
        self.session_id = session_id or self.id
        self.created_at = datetime.utcnow()
        self.last_active = self.created_at
        self.interactions = []
        self.emotions = []
        User.active_users[self.session_id] = self

def cleanup_old_sessions():
    """Clean up old sessions to free memory"""
    sessions = list(User.active_users.keys())
    current_time = datetime.now()
    for session_id in sessions:
        user = User.active_users[session_id]
        # Remove sessions older than 30 minutes
        if (current_time - user.last_active).total_seconds() > 1800:
            del User.active_users[session_id]

@app.before_request
def before_request():
    # Clean up old sessions every 100 requests
    if random.randint(1, 100) == 1:
        cleanup_old_sessions()
    
    # Create session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

# Modify get_or_create_user to use in-memory tracking
def get_or_create_user():
    if 'user_id' not in session:
        user = User()
        session['user_id'] = user.id
    else:
        user = User.active_users.get(session['user_id'])
        if user is None:
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

def get_ai_response(user_input, conversation_history):
    try:
        # Initialize or get user session
        if 'user_session' not in session:
            session['user_session'] = {}

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
        return render_template('index.html', initial_question=random.choice(ENGAGEMENT_QUESTIONS['initial']))
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = session.get('session_id')

        # Get AI response
        response = get_ai_response(user_message, session_id)
        
        # Create chat entry with minimal data
        chat_entry = {
            'session_id': session_id[:8],  # Only store first 8 chars of session ID
            'message': user_message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to chat history with size limit
        chat_history.append(chat_entry)
        if len(chat_history) > MAX_CHAT_HISTORY:
            chat_history.pop(0)
        
        # Emit to admin panel
        socketio.emit('new_chat', chat_entry, namespace='/admin')
        
        return jsonify({'response': response})

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    return render_template('admin/analytics.html', chat_history=chat_history)

# WebSocket events
@socketio.on('connect', namespace='/admin')
def handle_admin_connect():
    if not session.get('admin_logged_in'):
        return False
    emit('chat_history', chat_history)

if __name__ == '__main__':
    socketio.run(app, debug=True)
