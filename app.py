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
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_timeout=30,
    ping_interval=25,
    path='/socket.io',
    async_handlers=True,
    max_http_buffer_size=5e6  # 5MB max payload
)

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini AI model
model = genai.GenerativeModel('gemini-pro')

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

# Tone detection patterns
TONE_PATTERNS = {
    'casual_happy': [
        'haha', 'lol', 'xd', ':)', '😊', 'cool', 'awesome', 'nice', 'yay',
        'fun', 'great', 'amazing', 'love it', 'excited', 'wow'
    ],
    'formal_academic': [
        'study', 'research', 'theory', 'concept', 'psychology', 'understand',
        'explain', 'definition', 'example', 'analysis'
    ],
    'emotional_sad': [
        'sad', 'upset', 'crying', 'depressed', 'lonely', 'hurt', 'pain',
        'tired', 'exhausted', 'worried', 'anxious', '😢', '😔'
    ],
    'emotional_angry': [
        'angry', 'mad', 'frustrated', 'annoyed', 'hate', 'unfair', 
        'terrible', 'worst', '😠', '😡'
    ],
    'seeking_help': [
        'help', 'advice', 'suggestion', 'what should', 'how can i',
        'need to', 'struggling', 'difficult'
    ],
    'casual_friendly': [
        'hey', 'hi', 'hello', 'sup', 'whats up', 'how are you',
        'thanks', 'thank you', 'appreciate'
    ]
}

# Personality modes for different contexts
PERSONALITY_MODES = {
    'casual_happy': """Be cheerful and match their energy! Use casual language, emojis, and share their excitement.""",
    'formal_academic': """Be professional and informative. Use academic language and provide clear explanations with examples.""",
    'emotional_sad': """Be gentle and supportive. Use soft, comforting language and show deep empathy.""",
    'emotional_angry': """Be calm and understanding. Acknowledge frustration and help process emotions constructively.""",
    'seeking_help': """Be solution-focused. Listen actively and offer practical suggestions with encouragement.""",
    'casual_friendly': """Be warm and conversational. Use friendly language while maintaining professionalism."""
}

def detect_tone(message):
    """Detect the tone of the message based on keywords"""
    message_lower = message.lower()
    tone_scores = {}
    
    for tone, patterns in TONE_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern in message_lower)
        tone_scores[tone] = score
    
    # Get the dominant tone (highest score)
    dominant_tone = max(tone_scores.items(), key=lambda x: x[1])
    
    # If no clear tone is detected, default to casual_friendly
    if dominant_tone[1] == 0:
        return 'casual_friendly'
    
    return dominant_tone[0]

# Advanced personality and interaction patterns
PERSONALITY_TRAITS = {
    'openness': {
        'high': ['curious', 'creative', 'artistic', 'imaginative', 'innovative', 'deep', 'complex', 'philosophical'],
        'low': ['practical', 'conventional', 'straightforward', 'efficient', 'routine', 'simple']
    },
    'conscientiousness': {
        'high': ['organized', 'responsible', 'disciplined', 'efficient', 'planned', 'thorough', 'detailed'],
        'low': ['flexible', 'spontaneous', 'adaptable', 'easy-going', 'relaxed', 'casual']
    },
    'extraversion': {
        'high': ['outgoing', 'energetic', 'enthusiastic', 'social', 'talkative', 'expressive'],
        'low': ['quiet', 'reserved', 'reflective', 'thoughtful', 'calm', 'introspective']
    },
    'agreeableness': {
        'high': ['kind', 'sympathetic', 'cooperative', 'warm', 'considerate', 'friendly'],
        'low': ['direct', 'straightforward', 'honest', 'objective', 'frank']
    },
    'neuroticism': {
        'high': ['sensitive', 'emotional', 'intense', 'passionate', 'expressive'],
        'low': ['stable', 'calm', 'balanced', 'composed', 'steady']
    }
}

# Communication style patterns
COMMUNICATION_STYLES = {
    'assertive': {
        'keywords': ['need', 'want', 'think', 'believe', 'feel', 'prefer'],
        'response_style': 'Direct and confident while maintaining respect'
    },
    'analytical': {
        'keywords': ['why', 'how', 'what if', 'analyze', 'understand', 'explain'],
        'response_style': 'Logical and detailed with supporting information'
    },
    'expressive': {
        'keywords': ['amazing', 'awesome', 'love', 'hate', 'fantastic', 'terrible'],
        'response_style': 'Enthusiastic and emotionally engaging'
    },
    'supportive': {
        'keywords': ['help', 'support', 'please', 'thanks', 'appreciate', 'grateful'],
        'response_style': 'Warm and encouraging with validation'
    }
}

# Emotional intelligence patterns
EMOTIONAL_PATTERNS = {
    'primary_emotions': {
        'joy': ['happy', 'excited', 'delighted', 'pleased', 'glad', 'content'],
        'sadness': ['sad', 'down', 'upset', 'depressed', 'unhappy', 'blue'],
        'anger': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed'],
        'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
        'surprise': ['shocked', 'amazed', 'astonished', 'unexpected'],
        'disgust': ['disgusted', 'repulsed', 'revolted'],
        'trust': ['trust', 'believe', 'confident', 'faith', 'sure'],
        'anticipation': ['looking forward', 'excited about', 'cant wait', 'hoping']
    },
    'emotion_intensities': {
        'low': ['slightly', 'a bit', 'somewhat', 'mildly'],
        'medium': ['quite', 'rather', 'fairly', 'pretty'],
        'high': ['very', 'extremely', 'incredibly', 'absolutely']
    }
}

# Conversation context patterns
CONTEXT_PATTERNS = {
    'time_references': {
        'past': ['was', 'did', 'had', 'used to', 'before', 'previously'],
        'present': ['is', 'am', 'are', 'now', 'currently', 'these days'],
        'future': ['will', 'going to', 'planning to', 'hope to', 'want to']
    },
    'relationship_indicators': {
        'family': ['mom', 'dad', 'sister', 'brother', 'parent', 'family'],
        'friends': ['friend', 'buddy', 'pal', 'mate', 'colleague'],
        'romantic': ['partner', 'girlfriend', 'boyfriend', 'spouse', 'wife', 'husband'],
        'professional': ['boss', 'coworker', 'teacher', 'student', 'client']
    }
}

# Personality adaptation rules
ADAPTATION_RULES = {
    'matching': {
        'energy_level': 'Match the user\'s energy level in responses',
        'language_style': 'Mirror the user\'s vocabulary and sentence structure',
        'emotional_tone': 'Reflect the user\'s emotional state appropriately',
        'response_length': 'Match the typical length of user\'s messages'
    },
    'complementing': {
        'support_level': 'Provide more support when user shows vulnerability',
        'guidance_level': 'Offer more guidance when user seeks direction',
        'engagement_level': 'Increase engagement when user seems withdrawn'
    }
}

class PersonalityProfile:
    def __init__(self):
        self.traits = {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }
        self.communication_style = 'balanced'
        self.emotional_pattern = 'stable'
        self.interaction_history = []
        self.last_analysis = None
        self.analysis_cooldown = 3  # Only analyze every 3 messages
        
    def should_analyze(self):
        """Determine if we should perform full analysis"""
        if not self.last_analysis:
            return True
        return len(self.interaction_history) % self.analysis_cooldown == 0
    
    def quick_analysis(self, message):
        """Lightweight message analysis"""
        message_lower = message.lower()
        
        # Quick emotion check
        for emotion, keywords in EMOTIONAL_PATTERNS['primary_emotions'].items():
            if any(word in message_lower for word in keywords[:3]):  # Check only top keywords
                self.emotional_pattern = emotion
                break
        
        # Quick communication style check
        for style, data in COMMUNICATION_STYLES.items():
            if any(word in message_lower for word in data['keywords'][:3]):  # Check only top keywords
                self.communication_style = style
                break
    
    def update_traits(self, message):
        """Update personality traits based on message content"""
        message_lower = message.lower()
        
        # Analyze for each trait
        for trait, patterns in PERSONALITY_TRAITS.items():
            # Check high markers
            high_score = sum(1 for word in patterns['high'] if word in message_lower)
            # Check low markers
            low_score = sum(1 for word in patterns['low'] if word in message_lower)
            
            # Update trait score (weighted moving average)
            if high_score > 0 or low_score > 0:
                new_score = (high_score - low_score) / (high_score + low_score)
                self.traits[trait] = 0.8 * self.traits[trait] + 0.2 * ((new_score + 1) / 2)
    
    def analyze_communication_style(self, message):
        """Determine the dominant communication style"""
        message_lower = message.lower()
        style_scores = {}
        
        for style, data in COMMUNICATION_STYLES.items():
            score = sum(1 for keyword in data['keywords'] if keyword in message_lower)
            style_scores[style] = score
        
        # Get dominant style
        if any(style_scores.values()):
            self.communication_style = max(style_scores.items(), key=lambda x: x[1])[0]
    
    def analyze_emotional_pattern(self, message):
        """Analyze emotional patterns in the message"""
        message_lower = message.lower()
        emotion_scores = {}
        
        # Check primary emotions
        for emotion, keywords in EMOTIONAL_PATTERNS['primary_emotions'].items():
            score = sum(1 for word in keywords if word in message_lower)
            if score > 0:
                # Check intensity
                intensity = 'medium'
                for level, markers in EMOTIONAL_PATTERNS['emotion_intensities'].items():
                    if any(marker in message_lower for marker in markers):
                        intensity = level
                        break
                emotion_scores[emotion] = {'score': score, 'intensity': intensity}
        
        if emotion_scores:
            self.emotional_pattern = max(emotion_scores.items(), key=lambda x: x[1]['score'])[0]
    
    def update_profile(self, message, response):
        """Smart profile update with performance optimization"""
        # Always store interaction
        interaction = {
            'message': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        # Decide analysis depth
        if self.should_analyze():
            # Full analysis
            self.update_traits(message)
            self.analyze_communication_style(message)
            self.analyze_emotional_pattern(message)
            self.last_analysis = datetime.now()
            
            # Add detailed analysis to interaction
            interaction.update({
                'traits': self.traits.copy(),
                'style': self.communication_style,
                'emotion': self.emotional_pattern
            })
        else:
            # Quick analysis
            self.quick_analysis(message)
            interaction.update({
                'style': self.communication_style,
                'emotion': self.emotional_pattern
            })
        
        # Update history
        self.interaction_history.append(interaction)
        if len(self.interaction_history) > 10:
            self.interaction_history = self.interaction_history[-10:]

    def get_response_guidance(self):
        """Generate response guidance based on current profile"""
        guidance = {
            'style': COMMUNICATION_STYLES[self.communication_style]['response_style'],
            'traits': {
                trait: 'high' if score > 0.6 else 'low' if score < 0.4 else 'moderate'
                for trait, score in self.traits.items()
            },
            'emotion': self.emotional_pattern
        }
        return guidance

def get_ai_response(message, session_id):
    try:
        # Get or create user context
        user = User.active_users.get(session_id)
        if not user:
            user = User(session_id=session_id)
            user.personality_profile = PersonalityProfile()
        elif not hasattr(user, 'personality_profile'):
            user.personality_profile = PersonalityProfile()
        
        # Update last active time
        user.last_active = datetime.now()

        # Quick tone detection
        current_tone = detect_tone(message)
        
        # Get profile guidance (cached or quick analysis)
        profile_guidance = user.personality_profile.get_response_guidance()
        
        # Streamlined context
        context = f"""You are Nirya, an empathetic AI mental health companion.

        PROFILE:
        Tone: {current_tone}
        Style: {profile_guidance['style']}
        State: {profile_guidance['emotion']}
        
        KEY TRAITS:
        {', '.join(f'{t}:{l}' for t,l in list(profile_guidance['traits'].items())[:3])}

        Guidelines: Match style, show empathy, be natural."""

        # Efficient history handling
        history = ""
        if user.personality_profile.interaction_history:
            recent = user.personality_profile.interaction_history[-2:]  # Only last 2 interactions
            history = "\nRECENT:"
            for i in recent:
                history += f"\nU: {i['message']}\nN: {i['response']}"

        prompt = f"{context}\n{history}\n\nUser: {message}\n\nRespond:"

        # Get response from Gemini
        response = model.generate_content(prompt)
        ai_response = response.text.strip()
        
        # Update profile (with smart analysis)
        user.personality_profile.update_profile(message, ai_response)
        
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
