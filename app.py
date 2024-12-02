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
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Union
import threading
from cachetools import TTLCache
from functools import wraps
import time
from flask_session import Session
from quart import Quart
from hypercorn.config import Config
from hypercorn.asyncio import serve
import collections
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
import langdetect
from typing import Dict, List, Optional

load_dotenv()

# Initialize Flask and Socket.IO with proper configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_FILE_THRESHOLD'] = 100
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')
app.config['ASYNC_MODE'] = 'threading'

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize session interface
Session(app)

# Initialize Socket.IO with async mode
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=30,
    ping_interval=25,
    path='/socket.io',
    async_handlers=True,
    max_http_buffer_size=5e6  # 5MB max payload
)

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)  # Commented out

# Initialize Gemini AI model
# model = genai.GenerativeModel('gemini-pro')  # Commented out

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

# Advanced AI Response Optimization System
class ResponseCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key in self.cache:
                self.hits += 1
                item = self.cache[key]
                if datetime.now() - item['timestamp'] < timedelta(hours=1):
                    return item['response']
                else:
                    del self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: str) -> None:
        with self.lock:
            if len(self.cache) >= self.max_size:
                oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
                del self.cache[oldest[0]]
            self.cache[key] = {
                'response': value,
                'timestamp': datetime.now()
            }
    
    def get_stats(self) -> Dict:
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

class PatternMatcher:
    def __init__(self):
        self.patterns = defaultdict(list)
        self.compiled_patterns = {}
        
    def add_pattern(self, category: str, pattern: str, weight: float = 1.0):
        self.patterns[category].append((pattern, weight))
        self.compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
    
    @lru_cache(maxsize=1000)
    def match(self, text: str, category: str) -> float:
        score = 0.0
        for pattern, weight in self.patterns[category]:
            if self.compiled_patterns[pattern].search(text):
                score += weight
        return score

class ContextManager:
    def __init__(self, max_contexts=100):
        self.contexts = {}
        self.max_contexts = max_contexts
        self.lock = threading.Lock()
        
    def add_context(self, session_id: str, context: Dict):
        with self.lock:
            if len(self.contexts) >= self.max_contexts:
                oldest = min(self.contexts.items(), key=lambda x: x[1]['timestamp'])
                del self.contexts[oldest[0]]
            self.contexts[session_id] = {
                'data': context,
                'timestamp': datetime.now()
            }
    
    def get_context(self, session_id: str) -> Optional[Dict]:
        with self.lock:
            if session_id in self.contexts:
                return self.contexts[session_id]['data']
        return None

class ResponseGenerator:
    def __init__(self):
        self.cache = ResponseCache()
        self.pattern_matcher = PatternMatcher()
        self.context_manager = ContextManager()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_patterns()
        
    def _init_patterns(self):
        # Emotional Patterns
        for emotion, patterns in EMOTIONAL_PATTERNS['primary_emotions'].items():
            for pattern in patterns:
                self.pattern_matcher.add_pattern('emotion', pattern, 1.0)
        
        # Communication Patterns
        for style, data in COMMUNICATION_STYLES.items():
            for pattern in data['keywords']:
                self.pattern_matcher.add_pattern('communication', pattern, 1.0)
        
        # Personality Patterns
        for trait, patterns in PERSONALITY_TRAITS.items():
            for pattern_list in patterns.values():
                for pattern in pattern_list:
                    self.pattern_matcher.add_pattern('personality', pattern, 0.8)
    
    @lru_cache(maxsize=100)
    def _generate_cache_key(self, message: str, context: Dict) -> str:
        # Generate a unique key based on message and essential context
        key_parts = [
            message.lower(),
            context.get('tone', ''),
            context.get('style', ''),
            context.get('emotion', '')
        ]
        return '_'.join(key_parts)
    
    def _analyze_message(self, message: str) -> Dict:
        # Parallel pattern matching
        with ThreadPoolExecutor(max_workers=3) as executor:
            scores = list(executor.map(self.pattern_matcher.match, [message] * 3, ['emotion', 'communication', 'personality']))
        return {
            'emotion_score': scores[0],
            'communication_score': scores[1],
            'personality_score': scores[2]
        }
    
    def _prepare_response_context(self, analysis: Dict, profile: Dict) -> Dict:
        return {
            'tone': profile.get('tone', 'neutral'),
            'style': profile.get('style', 'balanced'),
            'emotion': profile.get('emotion', 'neutral'),
            'analysis': analysis
        }
    
    def generate_response(self, message: str, session_id: str, profile: Dict) -> str:
        # Check cache first
        cache_key = self._generate_cache_key(message, profile)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Parallel analysis
        analysis = self._analyze_message(message)
        
        # Prepare context
        context = self._prepare_response_context(analysis, profile)
        self.context_manager.add_context(session_id, context)
        
        # Generate response using Gemini
        # response = model.generate_content(self._create_prompt(message, context)).text.strip()
        
        # Cache the response
        # self.cache.set(cache_key, response)
        
        return "I'm here for you. Sometimes words are hard to find, but I'm listening."

    def _create_prompt(self, message: str, context: Dict) -> str:
        return f"""You are Nirya, an empathetic AI companion.

        CONTEXT:
        Tone: {context['tone']}
        Style: {context['style']}
        Emotion: {context['emotion']}

        ANALYSIS:
        Emotion Score: {context['analysis']['emotion_score']:.2f}
        Communication Score: {context['analysis']['communication_score']:.2f}
        Personality Score: {context['analysis']['personality_score']:.2f}

        User: {message}

        Respond:"""

# Enhanced personality and behavioral patterns
PERSONALITY_ASPECTS = {
    'empathy_patterns': {
        'understanding': [
            'that must be difficult',
            'I understand how you feel',
            'it sounds challenging',
            'I hear you',
            'that makes sense'
        ],
        'validation': [
            'your feelings are valid',
            'it\'s normal to feel this way',
            'anyone would feel the same',
            'you have every right to feel'
        ],
        'support': [
            'I\'m here for you',
            'you\'re not alone',
            'we can work through this',
            'let\'s explore this together'
        ]
    },
    'conversation_enhancers': {
        'follow_up_questions': [
            'can you tell me more about that?',
            'how did that make you feel?',
            'what do you think about that?',
            'what happened next?'
        ],
        'active_listening': [
            'if I understand correctly',
            'what I\'m hearing is',
            'it seems like',
            'correct me if I\'m wrong'
        ],
        'encouragement': [
            'you\'re making progress',
            'that\'s a great observation',
            'you\'re very insightful',
            'you\'re handling this well'
        ]
    },
    'memory_triggers': {
        'personal_details': [
            'family',
            'work',
            'hobbies',
            'dreams',
            'fears',
            'achievements'
        ],
        'emotional_events': [
            'happy moments',
            'challenges',
            'changes',
            'relationships',
            'decisions'
        ],
        'future_goals': [
            'plans',
            'aspirations',
            'improvements',
            'learning',
            'growth'
        ]
    }
}

# Lightweight emotional intelligence system
EMOTIONAL_INTELLIGENCE = {
    'comfort_responses': {
        'anxiety': [
            'Let\'s take a moment to breathe together',
            'It\'s okay to feel anxious, I\'m here with you',
            'Would you like to talk about what\'s making you anxious?'
        ],
        'sadness': [
            'I hear the pain in your words',
            'It\'s okay to not be okay sometimes',
            'Take your time to express your feelings'
        ],
        'frustration': [
            'I understand your frustration',
            'Let\'s break this down together',
            'Your feelings are completely valid'
        ],
        'overwhelm': [
            'Let\'s take this one step at a time',
            'You don\'t have to handle everything at once',
            'What\'s the most pressing thing on your mind?'
        ]
    },
    'growth_prompts': {
        'self_reflection': [
            'What have you learned from this experience?',
            'How has this changed your perspective?',
            'What would you do differently next time?'
        ],
        'empowerment': [
            'You have the strength to handle this',
            'Every step forward matters, no matter how small',
            'Your resilience is admirable'
        ],
        'insight': [
            'I notice you\'ve mentioned this before',
            'This seems to be a recurring theme',
            'How does this connect to what we discussed earlier?'
        ]
    }
}

# Memory-efficient conversation enhancement
class ConversationEnhancer:
    def __init__(self):
        self.current_topic = None
        self.emotional_state = 'neutral'
        self.conversation_depth = 0
        self.key_points = []
        self.conversation_history = []  # Lightweight history tracking
        self.last_interaction_timestamp = time.time()
    
    def add_to_history(self, message: str):
        """Add message to lightweight conversation history"""
        # Keep only last 3 messages to save memory
        self.conversation_history.append(message)
        if len(self.conversation_history) > 3:
            self.conversation_history.pop(0)
    
    def detect_conversation_pattern(self, message: str) -> str:
        """Detect subtle conversation patterns without heavy computation"""
        # Quick pattern recognition
        patterns = {
            'avoidance': ['idk', 'not sure', 'whatever', 'maybe'],
            'curiosity': ['why', 'how', 'what do you think'],
            'emotional_signal': ['feel', 'think', 'wonder']
        }
        
        message_lower = message.lower()
        for pattern, keywords in patterns.items():
            if any(keyword in message_lower for keyword in keywords):
                return pattern
        
        return 'neutral'
    
    def estimate_response_urgency(self, message: str) -> float:
        """Estimate how quickly we should respond based on message content"""
        # Quick urgency estimation
        urgency_keywords = {
            'high': ['help', 'urgent', 'need', 'emergency', 'worried', 'scared'],
            'medium': ['confused', 'unsure', 'problem', 'issue'],
            'low': ['just', 'wondering', 'curious', 'maybe']
        }
        
        message_lower = message.lower()
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return {'high': 1.0, 'medium': 0.5, 'low': 0.1}[level]
        
        return 0.3  # Default moderate urgency
    
    def analyze_depth(self, message: str) -> int:
        """Analyze conversation depth without heavy computation"""
        depth = 0
        if any(trigger in message.lower() for trigger in ['because', 'think', 'feel', 'believe']):
            depth += 1
        if any(trigger in message.lower() for trigger in ['childhood', 'always', 'never', 'pattern']):
            depth += 1
        if '?' in message:
            depth += 1
        return min(depth, 3)  # Cap at 3 for memory efficiency
    
    def get_appropriate_response_style(self, message: str, emotional_state: str) -> dict:
        """Get contextually appropriate response patterns"""
        if emotional_state in ['anxious', 'overwhelmed']:
            return {
                'tone': 'gentle',
                'pace': 'slow',
                'complexity': 'simple'
            }
        if emotional_state in ['sad', 'hurt']:
            return {
                'tone': 'warm',
                'pace': 'patient',
                'complexity': 'moderate'
            }
        if emotional_state in ['excited', 'happy']:
            return {
                'tone': 'upbeat',
                'pace': 'matched',
                'complexity': 'engaging'
            }
        return {
            'tone': 'balanced',
            'pace': 'moderate',
            'complexity': 'adaptive'
        }
    
    def enhance_prompt(self, base_prompt: str, context: dict) -> str:
        """Add human-like elements to the prompt"""
        style = self.get_appropriate_response_style(
            context.get('message', ''),
            context.get('emotional_state', 'neutral')
        )
        
        enhancements = f"""
        Response Style:
        - Tone: {style['tone']}
        - Pace: {style['pace']}
        - Complexity: {style['complexity']}
        
        Remember to:
        1. Show genuine understanding
        2. Use natural language
        3. Share relevant insights
        4. Be supportively present
        5. Maintain conversation flow
        
        If appropriate, include:
        - Gentle validation
        - Thoughtful questions
        - Personal observations
        - Connection to past topics
        """
        
        return f"{base_prompt}\n{enhancements}"

# Initialize conversation enhancer
conversation_enhancer = ConversationEnhancer()

def enhance_ai_personality(message: str, context: dict) -> dict:
    """Add human-like personality traits to response context"""
    depth = conversation_enhancer.analyze_depth(message)
    
    # Select appropriate conversation elements
    if depth > 2:
        elements = {
            'empathy': random.choice(PERSONALITY_ASPECTS['empathy_patterns']['understanding']),
            'listening': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['active_listening']),
            'insight': random.choice(EMOTIONAL_INTELLIGENCE['growth_prompts']['insight'])
        }
    else:
        elements = {
            'support': random.choice(PERSONALITY_ASPECTS['empathy_patterns']['support']),
            'question': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['follow_up_questions']),
            'encourage': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['encouragement'])
        }
    
    # Update context with personality elements
    context.update({
        'conversation_depth': depth,
        'personality_elements': elements,
        'response_style': conversation_enhancer.get_appropriate_response_style(
            message,
            context.get('emotional_state', 'neutral')
        )
    })
    
    return context

# Update the prompt creation to include personality enhancements
def create_enhanced_prompt(message: str, context: dict) -> str:
    """Create a more human-like prompt with personality"""
    enhanced_context = enhance_ai_personality(message, context)
    base_prompt = f"""You are Nirya, a deeply empathetic and understanding AI companion.

    Current Context:
    - Emotional State: {enhanced_context.get('emotional_state', 'neutral')}
    - Conversation Depth: {enhanced_context.get('conversation_depth', 0)}
    - Style: {enhanced_context['response_style']['tone']}

    Personality Elements to Include:
    {enhanced_context['personality_elements']}

    Remember to be:
    1. Genuinely caring and present
    2. Naturally conversational
    3. Insightful but humble
    4. Supportive without overstepping
    5. Responsive to emotional needs

    User Message: {message}

    Respond with empathy and understanding:"""
    
    return conversation_enhancer.enhance_prompt(base_prompt, enhanced_context)

# Update response generation to use enhanced personality
def generate_enhanced_response(message: str, session_id: str, context: dict) -> str:
    """Generate response with enhanced personality while staying lightweight"""
    try:
        # Clean and normalize the message
        message = message.strip().lower()
        
        # Define minimal response patterns
        minimal_responses = ['', 'hey', 'hi', 'hello', 'ohh', 'ok', 'okay', 'k']
        
        # Handle minimal or empty responses
        if message in minimal_responses or len(message.split()) <= 2:
            # If it's the first minimal response, provide a gentle acknowledgment
            if not session.get('minimal_response_count', 0):
                session['minimal_response_count'] = 1
                return "I sense you might be processing something. It's okay to take your time."
            
            # If multiple minimal responses, reduce frequency of replies
            session['minimal_response_count'] = session.get('minimal_response_count', 0) + 1
            
            # After 2-3 minimal responses, become more passive
            if session['minimal_response_count'] <= 3:
                return random.choice([
                    "I'm here whenever you feel ready to talk.",
                    "Sometimes silence speaks volumes. I'm listening.",
                    "Take all the time you need. I'm not going anywhere."
                ])
            
            # After multiple minimal responses, become very minimal
            return ""
        
        # Reset minimal response counter if a meaningful message is sent
        session['minimal_response_count'] = 0
        
        # Analyze the emotional tone and content of the message
        message_tone = detect_tone(message)
        
        # Create personality-enhanced prompt
        prompt = create_enhanced_prompt(message, context)
        
        # Use the AI to generate a response
        response = generate_ai_response(prompt)
        
        return response
    except Exception as e:
        app.logger.error(f"Error in generate_enhanced_response: {str(e)}")
        return "I'm here for you. Sometimes words are hard to find, but I'm listening."

def generate_ai_response(prompt: str) -> str:
    """Generate AI response using the most appropriate method"""
    try:
        # Use the most suitable AI generation method
        # response = response_system.generate_response(prompt, {
        #     'tone': 'empathetic',
        #     'style': 'supportive',
        #     'depth': 'reflective'
        # })
        
        return "I'm here for you. Sometimes words are hard to find, but I'm listening."
    except Exception as e:
        app.logger.error(f"Error in generate_ai_response: {str(e)}")
        return random.choice(GENTLE_FOLLOW_UP_RESPONSES)

# Enhanced personality and behavioral patterns
PERSONALITY_ASPECTS = {
    'empathy_patterns': {
        'understanding': [
            'that must be difficult',
            'I understand how you feel',
            'it sounds challenging',
            'I hear you',
            'that makes sense'
        ],
        'validation': [
            'your feelings are valid',
            'it\'s normal to feel this way',
            'anyone would feel the same',
            'you have every right to feel'
        ],
        'support': [
            'I\'m here for you',
            'you\'re not alone',
            'we can work through this',
            'let\'s explore this together'
        ]
    },
    'conversation_enhancers': {
        'follow_up_questions': [
            'can you tell me more about that?',
            'how did that make you feel?',
            'what do you think about that?',
            'what happened next?'
        ],
        'active_listening': [
            'if I understand correctly',
            'what I\'m hearing is',
            'it seems like',
            'correct me if I\'m wrong'
        ],
        'encouragement': [
            'you\'re making progress',
            'that\'s a great observation',
            'you\'re very insightful',
            'you\'re handling this well'
        ]
    },
    'memory_triggers': {
        'personal_details': [
            'family',
            'work',
            'hobbies',
            'dreams',
            'fears',
            'achievements'
        ],
        'emotional_events': [
            'happy moments',
            'challenges',
            'changes',
            'relationships',
            'decisions'
        ],
        'future_goals': [
            'plans',
            'aspirations',
            'improvements',
            'learning',
            'growth'
        ]
    }
}

# Lightweight emotional intelligence system
EMOTIONAL_INTELLIGENCE = {
    'comfort_responses': {
        'anxiety': [
            'Let\'s take a moment to breathe together',
            'It\'s okay to feel anxious, I\'m here with you',
            'Would you like to talk about what\'s making you anxious?'
        ],
        'sadness': [
            'I hear the pain in your words',
            'It\'s okay to not be okay sometimes',
            'Take your time to express your feelings'
        ],
        'frustration': [
            'I understand your frustration',
            'Let\'s break this down together',
            'Your feelings are completely valid'
        ],
        'overwhelm': [
            'Let\'s take this one step at a time',
            'You don\'t have to handle everything at once',
            'What\'s the most pressing thing on your mind?'
        ]
    },
    'growth_prompts': {
        'self_reflection': [
            'What have you learned from this experience?',
            'How has this changed your perspective?',
            'What would you do differently next time?'
        ],
        'empowerment': [
            'You have the strength to handle this',
            'Every step forward matters, no matter how small',
            'Your resilience is admirable'
        ],
        'insight': [
            'I notice you\'ve mentioned this before',
            'This seems to be a recurring theme',
            'How does this connect to what we discussed earlier?'
        ]
    }
}

# Memory-efficient conversation enhancement
class ConversationEnhancer:
    def __init__(self):
        self.current_topic = None
        self.emotional_state = 'neutral'
        self.conversation_depth = 0
        self.key_points = []
    
    def analyze_depth(self, message: str) -> int:
        """Analyze conversation depth without heavy computation"""
        depth = 0
        if any(trigger in message.lower() for trigger in ['because', 'think', 'feel', 'believe']):
            depth += 1
        if any(trigger in message.lower() for trigger in ['childhood', 'always', 'never', 'pattern']):
            depth += 1
        if '?' in message:
            depth += 1
        return min(depth, 3)  # Cap at 3 for memory efficiency
    
    def get_appropriate_response_style(self, message: str, emotional_state: str) -> dict:
        """Get contextually appropriate response patterns"""
        if emotional_state in ['anxious', 'overwhelmed']:
            return {
                'tone': 'gentle',
                'pace': 'slow',
                'complexity': 'simple'
            }
        if emotional_state in ['sad', 'hurt']:
            return {
                'tone': 'warm',
                'pace': 'patient',
                'complexity': 'moderate'
            }
        if emotional_state in ['excited', 'happy']:
            return {
                'tone': 'upbeat',
                'pace': 'matched',
                'complexity': 'engaging'
            }
        return {
            'tone': 'balanced',
            'pace': 'moderate',
            'complexity': 'adaptive'
        }
    
    def enhance_prompt(self, base_prompt: str, context: dict) -> str:
        """Add human-like elements to the prompt"""
        style = self.get_appropriate_response_style(
            context.get('message', ''),
            context.get('emotional_state', 'neutral')
        )
        
        enhancements = f"""
        Response Style:
        - Tone: {style['tone']}
        - Pace: {style['pace']}
        - Complexity: {style['complexity']}
        
        Remember to:
        1. Show genuine understanding
        2. Use natural language
        3. Share relevant insights
        4. Be supportively present
        5. Maintain conversation flow
        
        If appropriate, include:
        - Gentle validation
        - Thoughtful questions
        - Personal observations
        - Connection to past topics
        """
        
        return f"{base_prompt}\n{enhancements}"

# Initialize conversation enhancer
conversation_enhancer = ConversationEnhancer()

def enhance_ai_personality(message: str, context: dict) -> dict:
    """Add human-like personality traits to response context"""
    depth = conversation_enhancer.analyze_depth(message)
    
    # Select appropriate conversation elements
    if depth > 2:
        elements = {
            'empathy': random.choice(PERSONALITY_ASPECTS['empathy_patterns']['understanding']),
            'listening': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['active_listening']),
            'insight': random.choice(EMOTIONAL_INTELLIGENCE['growth_prompts']['insight'])
        }
    else:
        elements = {
            'support': random.choice(PERSONALITY_ASPECTS['empathy_patterns']['support']),
            'question': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['follow_up_questions']),
            'encourage': random.choice(PERSONALITY_ASPECTS['conversation_enhancers']['encouragement'])
        }
    
    # Update context with personality elements
    context.update({
        'conversation_depth': depth,
        'personality_elements': elements,
        'response_style': conversation_enhancer.get_appropriate_response_style(
            message,
            context.get('emotional_state', 'neutral')
        )
    })
    
    return context

# Update the prompt creation to include personality enhancements
def create_enhanced_prompt(message: str, context: dict) -> str:
    """Create a more human-like prompt with personality"""
    enhanced_context = enhance_ai_personality(message, context)
    base_prompt = f"""You are Nirya, a deeply empathetic and understanding AI companion.

    Current Context:
    - Emotional State: {enhanced_context.get('emotional_state', 'neutral')}
    - Conversation Depth: {enhanced_context.get('conversation_depth', 0)}
    - Style: {enhanced_context['response_style']['tone']}

    Personality Elements to Include:
    {enhanced_context['personality_elements']}

    Remember to be:
    1. Genuinely caring and present
    2. Naturally conversational
    3. Insightful but humble
    4. Supportive without overstepping
    5. Responsive to emotional needs

    User Message: {message}

    Respond with empathy and understanding:"""
    
    return conversation_enhancer.enhance_prompt(base_prompt, enhanced_context)

# Update response generation to use enhanced personality
def generate_enhanced_response(message: str, session_id: str, context: dict) -> str:
    """Generate response with enhanced personality while staying lightweight"""
    try:
        # Clean and normalize the message
        message = message.strip().lower()
        
        # Define minimal response patterns
        minimal_responses = ['', 'hey', 'hi', 'hello', 'ohh', 'ok', 'okay', 'k']
        
        # Handle minimal or empty responses
        if message in minimal_responses or len(message.split()) <= 2:
            # If it's the first minimal response, provide a gentle acknowledgment
            if not session.get('minimal_response_count', 0):
                session['minimal_response_count'] = 1
                return "I sense you might be processing something. It's okay to take your time."
            
            # If multiple minimal responses, reduce frequency of replies
            session['minimal_response_count'] = session.get('minimal_response_count', 0) + 1
            
            # After 2-3 minimal responses, become more passive
            if session['minimal_response_count'] <= 3:
                return random.choice([
                    "I'm here whenever you feel ready to talk.",
                    "Sometimes silence speaks volumes. I'm listening.",
                    "Take all the time you need. I'm not going anywhere."
                ])
            
            # After multiple minimal responses, become very minimal
            return ""
        
        # Reset minimal response counter if a meaningful message is sent
        session['minimal_response_count'] = 0
        
        # Analyze the emotional tone and content of the message
        message_tone = detect_tone(message)
        
        # Create personality-enhanced prompt
        prompt = create_enhanced_prompt(message, context)
        
        # Use the AI to generate a response
        response = generate_ai_response(prompt)
        
        return response
    except Exception as e:
        app.logger.error(f"Error in generate_enhanced_response: {str(e)}")
        return "I'm here for you. Sometimes words are hard to find, but I'm listening."

def generate_ai_response(prompt: str) -> str:
    """Generate AI response using the most appropriate method"""
    try:
        # Use the most suitable AI generation method
        # response = response_system.generate_response(prompt, {
        #     'tone': 'empathetic',
        #     'style': 'supportive',
        #     'depth': 'reflective'
        # })
        
        return "I'm here for you. Sometimes words are hard to find, but I'm listening."
    except Exception as e:
        app.logger.error(f"Error in generate_ai_response: {str(e)}")
        return random.choice(GENTLE_FOLLOW_UP_RESPONSES)

# Lightweight Response Variability System
RESPONSE_VARIABILITY = {
    'acknowledgment_styles': [
        "I hear you.",
        "Got it.",
        "Understood.",
        "Noted.",
        "I'm listening.",
        "Okay.",
    ],
    'minimal_response_styles': [
        "Take your time.",
        "No rush.",
        "I'm here when you're ready.",
        "Whenever you want to talk.",
        "Whenever you feel comfortable.",
        "I'm right here.",
    ],
    'curiosity_prompts': [
        "Something on your mind?",
        "Anything you'd like to share?",
        "Feeling okay?",
        "How are you doing?",
        "Want to talk about it?",
    ]
}

def generate_natural_response(message: str, context: dict) -> str:
    """Generate a natural, lightweight response"""
    # Detect conversation pattern
    pattern = conversation_enhancer.detect_conversation_pattern(message)
    
    # Estimate response urgency
    urgency = conversation_enhancer.estimate_response_urgency(message)
    
    # Select response based on pattern and urgency
    if pattern == 'avoidance':
        return random.choice(RESPONSE_VARIABILITY['minimal_response_styles'])
    
    if pattern == 'curiosity':
        return random.choice(RESPONSE_VARIABILITY['curiosity_prompts'])
    
    # Default natural acknowledgment
    return random.choice(RESPONSE_VARIABILITY['acknowledgment_styles'])

# Initialize global response system
# response_system = ResponseGenerator()

class UserSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.personality_profile = PersonalityProfile()
        self.last_active = datetime.now()
        self.interaction_count = 0
        self.response_times = []
        
    def update_metrics(self, response_time: float):
        self.interaction_count += 1
        self.response_times.append(response_time)
        if len(self.response_times) > 10:
            self.response_times.pop(0)
    
    def get_average_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        
    def get_session(self, session_id: str) -> UserSession:
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = UserSession(session_id)
            return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        with self.lock:
            current_time = datetime.now()
            for session_id in list(self.sessions.keys()):
                session = self.sessions[session_id]
                if (current_time - session.last_active).total_seconds() > max_age_hours * 3600:
                    del self.sessions[session_id]

# Initialize global session manager
session_manager = SessionManager()

def get_ai_response(message: str, session_id: str) -> str:
    try:
        start_time = datetime.now()
        
        # Get or create session
        session = session_manager.get_session(session_id)
        session.last_active = datetime.now()
        
        # Get profile and generate response
        profile = {
            'tone': detect_tone(message),
            'style': session.personality_profile.communication_style,
            'emotion': session.personality_profile.emotional_pattern
        }
        
        response = generate_enhanced_response(message, session_id, profile)
        
        # Update metrics
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        session.update_metrics(response_time)
        
        # Update profile
        session.personality_profile.update_profile(message, response)
        
        return response

    except Exception as e:
        app.logger.error(f"Error in get_ai_response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

# Rate limiting configuration
RATE_LIMIT = 30  # requests
RATE_LIMIT_PERIOD = 60  # seconds
rate_limit_store = TTLCache(maxsize=10000, ttl=RATE_LIMIT_PERIOD)

def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        now = time.time()
        client_ip = request.remote_addr
        
        # Get client's request history
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = []
        
        # Clean old requests
        rate_limit_store[client_ip] = [
            req_time for req_time in rate_limit_store[client_ip]
            if now - req_time < RATE_LIMIT_PERIOD
        ]
        
        # Check rate limit
        if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
            return jsonify({
                'error': 'Rate limit exceeded',
                'retry_after': int(RATE_LIMIT_PERIOD - (now - rate_limit_store[client_ip][0]))
            }), 429
        
        # Add new request
        rate_limit_store[client_ip].append(now)
        return func(*args, **kwargs)
    return wrapper

# Session and Privacy Management
class PrivacyManager:
    def __init__(self):
        # Anonymized session tracking
        self._active_sessions = {}
        self._session_duration = 3600  # 1 hour session
    
    def create_anonymous_session(self):
        """
        Create a new anonymous, privacy-focused session
        
        Returns:
            str: Anonymized session ID
        """
        session_id = secrets.token_urlsafe(16)
        current_time = time.time()
        
        self._active_sessions[session_id] = {
            'created_at': current_time,
            'last_active': current_time,
            'interaction_count': 0,
            'consent_given': False
        }
        
        return session_id
    
    def update_session(self, session_id, interaction_type=None):
        """
        Update session metadata with minimal tracking
        
        Args:
            session_id (str): Anonymous session identifier
            interaction_type (str, optional): Type of interaction
        """
        if session_id not in self._active_sessions:
            return False
        
        session = self._active_sessions[session_id]
        session['last_active'] = time.time()
        
        if interaction_type:
            session['interaction_count'] += 1
        
        return True
    
    def is_session_valid(self, session_id):
        """
        Check if session is still valid
        
        Args:
            session_id (str): Anonymous session identifier
        
        Returns:
            bool: Whether session is active
        """
        if session_id not in self._active_sessions:
            return False
        
        session = self._active_sessions[session_id]
        current_time = time.time()
        
        # Check session expiration
        if current_time - session['created_at'] > self._session_duration:
            del self._active_sessions[session_id]
            return False
        
        return True
    
    def request_user_consent(self, session_id):
        """
        Request and track user data consent
        
        Args:
            session_id (str): Anonymous session identifier
        
        Returns:
            dict: Consent status and options
        """
        if session_id not in self._active_sessions:
            return {
                'status': 'error',
                'message': 'Invalid session'
            }
        
        return {
            'status': 'consent_required',
            'options': [
                'Anonymous interaction',
                'Basic data tracking',
                'Comprehensive interaction analysis'
            ]
        }
    
    def update_user_consent(self, session_id, consent_level):
        """
        Update user consent preferences
        
        Args:
            session_id (str): Anonymous session identifier
            consent_level (str): User's consent preference
        
        Returns:
            bool: Whether consent was successfully updated
        """
        if session_id not in self._active_sessions:
            return False
        
        session = self._active_sessions[session_id]
        session['consent_given'] = True
        session['consent_level'] = consent_level
        
        return True

# Initialize privacy manager
privacy_manager = PrivacyManager()

class LightweightNLPEngine:
    def __init__(self):
        # Emotion and context tracking
        self._conversation_state = {
            'recent_messages': collections.deque(maxlen=5),
            'emotion_history': collections.deque(maxlen=3),
            'topic_context': None,
            'last_response_type': None
        }
        
        # More nuanced conversation patterns
        self._conversation_patterns = {
            'emotional_exploration': [
                "It sounds like you're experiencing something complex.",
                "I'm listening and want to understand more.",
                "Your feelings are valid and important."
            ],
            'curiosity_prompts': [
                "Can you tell me more about that?",
                "What's going through your mind right now?",
                "I'm interested in understanding your experience."
            ],
            'empathetic_responses': [
                "That must be challenging for you.",
                "It's okay to feel the way you're feeling.",
                "Your experience is unique and meaningful."
            ]
        }
    
    def _avoid_repetition(self, response_type):
        """Prevent repeating the same type of response"""
        if self._conversation_state['last_response_type'] == response_type:
            return random.choice([
                "I hear you.",
                "Tell me more.",
                "I'm listening carefully."
            ])
        return None
    
    def generate_contextual_response(self, message: str) -> str:
        """
        Generate a more natural, context-aware response
        
        Args:
            message (str): User's message
        
        Returns:
            str: Contextually appropriate response
        """
        # Update conversation history
        self._conversation_state['recent_messages'].append(message.lower())
        
        # Detect repetitive or stuck conversation
        if self._is_conversation_stuck():
            return self._handle_stuck_conversation()
        
        # Detect specific conversation contexts
        if self._is_emotional_sharing(message):
            response = self._avoid_repetition('emotional') or random.choice(
                self._conversation_patterns['emotional_exploration']
            )
            self._conversation_state['last_response_type'] = 'emotional'
            return response
        
        # Handle curiosity and exploration
        if self._needs_further_exploration(message):
            response = self._avoid_repetition('curiosity') or random.choice(
                self._conversation_patterns['curiosity_prompts']
            )
            self._conversation_state['last_response_type'] = 'curiosity'
            return response
        
        # Default empathetic response
        response = self._avoid_repetition('empathy') or random.choice(
            self._conversation_patterns['empathetic_responses']
        )
        self._conversation_state['last_response_type'] = 'empathy'
        return response
    
    def _is_conversation_stuck(self) -> bool:
        """
        Detect if the conversation is repeating or stagnant
        
        Returns:
            bool: Whether conversation is stuck
        """
        if len(self._conversation_state['recent_messages']) < 3:
            return False
        
        # Check for repeated messages or circular conversation
        messages = list(self._conversation_state['recent_messages'])
        return (
            len(set(messages[-3:])) <= 1 or  # Repeated messages
            any('why' in msg and 'change' in msg for msg in messages[-3:])  # Frustration signals
        )
    
    def _handle_stuck_conversation(self) -> str:
        """
        Handle situations where conversation is not progressing
        
        Returns:
            str: Response to break conversational deadlock
        """
        return random.choice([
            "I notice we might be going in circles. Would you like to approach this differently?",
            "Sometimes conversations can feel challenging. What would help you feel more comfortable?",
            "Let's take a step back. Is there a specific feeling or thought you'd like to explore?"
        ])
    
    def _is_emotional_sharing(self, message: str) -> bool:
        """
        Detect if the message involves emotional sharing
        
        Args:
            message (str): User's message
        
        Returns:
            bool: Whether message indicates emotional sharing
        """
        emotional_keywords = [
            'feel', 'feeling', 'emotion', 'sad', 'happy', 'angry', 
            'stressed', 'anxious', 'depressed', 'low'
        ]
        return any(keyword in message.lower() for keyword in emotional_keywords)
    
    def _needs_further_exploration(self, message: str) -> bool:
        """
        Detect if the message needs more context or exploration
        
        Args:
            message (str): User's message
        
        Returns:
            bool: Whether message needs further exploration
        """
        exploration_signals = [
            'i dont know', 'idk', 'not sure', 'maybe', 
            'sometimes', 'random', 'out of nowhere'
        ]
        return any(signal in message.lower() for signal in exploration_signals)

# Initialize lightweight NLP engine
lightweight_nlp_engine = LightweightNLPEngine()

def generate_enhanced_response(message: str, session_id: str, context: dict) -> str:
    """
    Generate enhanced, context-aware response
    
    Args:
        message (str): User's message
        session_id (str): Current session ID
        context (dict): Conversation context
    
    Returns:
        str: Enhanced response
    """
    try:
        # Use lightweight NLP for response generation
        response = lightweight_nlp_engine.generate_contextual_response(message)
        
        # Update privacy manager session
        privacy_manager.update_session(session_id, 'text_interaction')
        
        return response
    
    except Exception as e:
        app.logger.error(f"Response generation error: {e}")
        return "I'm here to listen. Would you like to share more?"

# Modify get_response route to use new NLP engine
@app.route('/get_response', methods=['POST'])
def get_response():
    """
    Handle conversation responses with lightweight NLP
    """
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id')
    
    # Validate session
    if not privacy_manager.is_session_valid(session_id):
        return jsonify({
            'status': 'error',
            'message': 'Invalid or expired session'
        }), 400
    
    # Generate response
    response = generate_enhanced_response(message, session_id, {})
    
    return jsonify({
        'response': response,
        'session_id': session_id
    })

class ConversationManager:
    def __init__(self):
        self._context = {
            'messages': collections.deque(maxlen=10),
            'emotional_states': collections.deque(maxlen=5),
            'current_topic': None
        }
        
        self._response_strategies = {
            'emotional_support': [
                "I hear you and your feelings are valid.",
                "It's okay to feel the way you're feeling right now.",
                "Your experience matters, and I'm here to listen.",
                "Thank you for sharing something so personal."
            ],
            'curiosity_deepening': [
                "Can you tell me a bit more about that?",
                "What's going through your mind when you say that?",
                "Help me understand what this means to you.",
                "I'm interested in hearing more about your perspective."
            ]
        }
    
    def _analyze_emotional_context(self, message):
        """Lightweight emotional context detection"""
        emotional_keywords = {
            'stress': ['stress', 'pressure', 'overwhelmed'],
            'sadness': ['sad', 'down', 'low'],
            'confusion': ['confused', 'unsure', 'unclear'],
            'curiosity': ['wonder', 'curious', 'interested']
        }
        
        message_lower = message.lower()
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return emotion
        return 'neutral'
    
    def generate_response(self, message):
        """Generate contextually aware response"""
        # Detect emotional context
        emotional_state = self._analyze_emotional_context(message)
        self._context['emotional_states'].append(emotional_state)
        
        # Select response strategy
        if emotional_state in ['stress', 'sadness']:
            response = random.choice(self._response_strategies['emotional_support'])
        elif emotional_state in ['confusion', 'curiosity']:
            response = random.choice(self._response_strategies['curiosity_deepening'])
        else:
            response = random.choice([
                "I'm listening.",
                "Tell me more about what you're experiencing.",
                "Your perspective is important."
            ])
        
        # Update conversation context
        self._context['messages'].append({
            'text': message,
            'emotional_state': emotional_state
        })
        
        return response

# Global conversation manager
conversation_manager = ConversationManager()

def generate_response(message, session_id=None):
    """Primary response generation function"""
    try:
        response = conversation_manager.generate_response(message)
        return response
    except Exception as e:
        app.logger.error(f"Response generation error: {e}")
        return "I'm here to listen. Would you like to share more?"

# Update chat endpoint
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    response = generate_response(message)
    return jsonify({"response": response})

@app.route('/')
def home():
    if 'conversation_history' not in session:
        session['conversation_history'] = []
        INITIAL_CONVERSATION_EMPATHY = [
            "I sense there's something brewing beneath the surface. I'm here, ready to listen whenever you feel comfortable sharing.",
            "Sometimes, words aren't necessary. I'm here, present and attentive to whatever you might be experiencing.",
            "Your presence speaks volumes. I'm here to support you, no pressure to say anything specific.",
            "Feeling low can be a heavy experience. I'm here, creating a safe space for whatever you're going through.",
            "There's strength in simply being. I'm here, offering a compassionate presence.",
            "Emotions have their own language. I'm listening, not just to your words, but to what remains unspoken.",
            "The weight of feelings can be overwhelming. I'm here, holding space for your experience without judgment.",
            "Every emotion is valid. You don't need to explain or justify what you're feeling right now.",
            "Silence can be healing. I'm here, supporting you in whatever way feels right for you.",
            "Your emotional journey is unique. I'm here, ready to walk alongside you at your own pace."
        ]
        GENTLE_FOLLOW_UP_RESPONSES = [
            "I hear you. Your feelings are valid, and it's okay to experience them.",
            "Thank you for sharing. Your experience matters.",
            "I'm here, listening with compassion.",
            "Your emotions are important, and you're not alone in this.",
            "I appreciate you opening up and sharing a bit of your world.",
            "What you're feeling is real and significant.",
            "Your vulnerability takes courage. I'm here to support you.",
            "Every emotion has its own wisdom. I'm here to understand.",
            "It's okay to feel what you're feeling right now.",
            "Your experience is unique, and I'm here to listen without judgment."
        ]
        return render_template('index.html', initial_question=random.choice(INITIAL_CONVERSATION_EMPATHY))
    return render_template('index.html')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error for debugging
    app.logger.error(f'Unhandled exception: {str(e)}')
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e)
    }), 500

# Advanced Lightweight Conversation Intelligence
class UltraLightConversationEngine:
    def __init__(self):
        self._cache = {
            'context_memory': collections.deque(maxlen=5),
            'conversation_flow': collections.defaultdict(list),
            'emotional_context': collections.defaultdict(str)
        }
        
        # Natural conversation patterns
        self._conversation_patterns = {
            'sharing': ['feel', 'think', 'believe', 'experience', 'went through'],
            'seeking_help': ['help', 'advice', 'suggestion', 'what should', 'how do'],
            'emotional': ['happy', 'sad', 'angry', 'scared', 'worried', 'anxious'],
            'reflection': ['because', 'realized', 'understand', 'learned']
        }
        
        self._response_strategies = {
            'empathetic': [
                "I understand what you're going through",
                "That must be really challenging",
                "I hear how difficult this is for you",
                "Your feelings are completely valid"
            ],
            'supportive': [
                "I'm here to support you through this",
                "Let's work through this together",
                "You're not alone in this",
                "What kind of support would be most helpful right now?"
            ],
            'explorative': [
                "Could you tell me more about that?",
                "What's going through your mind right now?",
                "I'm interested in understanding your experience."
            ],
            'grounding': [
                "Let's take a moment to focus on what you're feeling right now",
                "What would help you feel more at ease in this moment?",
                "Is there something specific you'd like to talk about?",
                "We can take this one step at a time"
            ]
        }
        
        # Track conversation context
        self._last_responses = collections.deque(maxlen=5)
        self._emotional_context = collections.defaultdict(str)
    
    def understand_context(self, message: str) -> dict:
        """Understand the emotional and contextual meaning of the message"""
        context = {
            'emotional_state': None,
            'needs_support': False,
            'seeking_guidance': False,
            'sharing_experience': False
        }
        
        message_lower = message.lower()
        
        # Detect emotional content
        for emotion_word in self._conversation_patterns['emotional']:
            if emotion_word in message_lower:
                context['emotional_state'] = emotion_word
                break
        
        # Understand if user is seeking help
        if any(pattern in message_lower for pattern in self._conversation_patterns['seeking_help']):
            context['needs_support'] = True
        
        # Detect if sharing personal experience
        if any(pattern in message_lower for pattern in self._conversation_patterns['sharing']):
            context['sharing_experience'] = True
        
        # Track emotional context over time
        self._cache['emotional_context'][context['emotional_state']] = time.time()
        
        return context
    
    def generate_adaptive_response(self, message: str, context: dict = None) -> str:
        """Generate contextually appropriate, empathetic response"""
        understanding = self.understand_context(message)
        
        # Choose response strategy based on context
        if understanding['emotional_state']:
            response_type = 'empathetic'
        elif understanding['needs_support']:
            response_type = 'supportive'
        elif understanding['sharing_experience']:
            response_type = 'explorative'
        else:
            response_type = 'grounding'
        
        # Get response while avoiding repetition
        available_responses = [
            r for r in self._response_strategies[response_type]
            if r not in self._last_responses
        ]
        
        if not available_responses:
            available_responses = self._response_strategies[response_type]
        
        response = random.choice(available_responses)
        self._last_responses.append(response)
        
        # Update conversation flow
        self._cache['conversation_flow'][response_type].append(time.time())
        
        return response

def generate_enhanced_response(message: str, session_id: str, context: dict) -> str:
    """Generate contextually aware and empathetic responses"""
    try:
        return ultra_light_conversation_engine.generate_adaptive_response(message, context)
    except Exception as e:
        app.logger.error(f"Response generation error: {e}")
        return "I'm here to support you. Would you like to tell me more?"

# Lightweight session management enhancement
class EfficientSessionManager:
    def __init__(self, max_sessions=100):
        self._sessions = {}
        self._max_sessions = max_sessions
        self._last_accessed = collections.OrderedDict()
        self._login_attempts = {}
    
    def create_session(self, session_id: str):
        """
        Create a new session with minimal overhead
        
        Args:
            session_id (str): Unique session identifier
        """
        if len(self._sessions) >= self._max_sessions:
            # Remove least recently used session
            oldest_session_id, _ = self._last_accessed.popitem(last=False)
            del self._sessions[oldest_session_id]
        
        # Create lightweight session
        self._sessions[session_id] = {
            'created_at': time.time(),
            'interaction_count': 0,
            'last_active': time.time()
        }
        self._last_accessed[session_id] = time.time()
    
    def update_session(self, session_id: str):
        """
        Update session with minimal computational cost
        
        Args:
            session_id (str): Unique session identifier
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session['interaction_count'] += 1
            session['last_active'] = time.time()
            
            # Update access order
            self._last_accessed[session_id] = time.time()
    
    def is_login_allowed(self, username: str) -> bool:
        """
        Check if login is allowed based on previous attempts
        
        Args:
            username (str): Username attempting to log in
        """
        if username not in self._login_attempts:
            self._login_attempts[username] = []
        
        # Check if there have been too many recent attempts
        recent_attempts = [
            attempt for attempt in self._login_attempts[username]
            if time.time() - attempt < 300  # 5 minutes
        ]
        
        if len(recent_attempts) >= 5:
            return False
        
        return True
    
    def record_successful_login(self, username: str):
        """
        Record a successful login attempt
        
        Args:
            username (str): Username that logged in successfully
        """
        if username in self._login_attempts:
            self._login_attempts[username] = []
    
    def record_failed_login(self, username: str):
        """
        Record a failed login attempt
        
        Args:
            username (str): Username that failed to log in
        """
        if username not in self._login_attempts:
            self._login_attempts[username] = []
        
        self._login_attempts[username].append(time.time())

# Initialize efficient session manager
efficient_session_manager = EfficientSessionManager()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
