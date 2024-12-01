from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import os
from dotenv import load_dotenv
from functools import wraps
import hashlib
import random
import json
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
import openai

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

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_ai_response(message, session_id):
    try:
        # Get user context
        user = User.active_users.get(session_id)
        if not user:
            user = User(session_id=session_id)
        
        # Update last active time
        user.last_active = datetime.now()
        
        # Create a system message that defines the AI's role and personality
        system_message = """You are Nirya, an empathetic AI mental health companion. Your responses should be:
        1. Warm and supportive, but professional
        2. Focused on the user's emotional well-being
        3. Consistent with previous responses
        4. Educational when discussing psychology topics
        5. Clear and direct when answering academic questions
        
        Important guidelines:
        - If asked about psychology topics, provide accurate academic information
        - If the user is sharing personal experiences, focus on emotional support
        - Maintain conversation context and avoid generic responses
        - If you don't know something, admit it honestly
        - Keep responses concise but meaningful"""

        # Create conversation history
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]

        # Add recent chat history for context (last 3 messages)
        if hasattr(user, 'chat_history') and user.chat_history:
            recent_history = user.chat_history[-3:]
            for chat in recent_history:
                conversation.extend([
                    {"role": "user", "content": chat['message']},
                    {"role": "assistant", "content": chat['response']}
                ])

        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=200,  # Limit response length
            temperature=0.7,  # Balance between creativity and consistency
            presence_penalty=0.6,  # Encourage topic variation
            frequency_penalty=0.3  # Reduce repetition
        )

        # Extract and store the response
        ai_response = response.choices[0].message['content'].strip()
        
        # Update user's chat history
        if not hasattr(user, 'chat_history'):
            user.chat_history = []
        user.chat_history.append({
            'message': message,
            'response': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 5 messages in user history
        if len(user.chat_history) > 5:
            user.chat_history = user.chat_history[-5:]

        return ai_response

    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

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
        return render_template('index.html', initial_question=random.choice(["I'm here to listen and support you. What brings you here today?", "Before we begin, how are you feeling right now? Take your time to share.", "I'd love to understand what's been going on for you lately. What would you like to explore?", "Thank you for reaching out. What's been on your mind that you'd like to discuss?"]))
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
