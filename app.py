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
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Union
import threading

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
    
    async def _analyze_message_async(self, message: str) -> Dict:
        # Parallel pattern matching
        tasks = []
        for category in ['emotion', 'communication', 'personality']:
            tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.pattern_matcher.match,
                    message,
                    category
                )
            )
        scores = await asyncio.gather(*tasks)
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
    
    async def generate_response(self, message: str, session_id: str, profile: Dict) -> str:
        # Check cache first
        cache_key = self._generate_cache_key(message, profile)
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Parallel analysis
        analysis = await self._analyze_message_async(message)
        
        # Prepare context
        context = self._prepare_response_context(analysis, profile)
        self.context_manager.add_context(session_id, context)
        
        # Generate response using Gemini
        response = model.generate_content(self._create_prompt(message, context)).text.strip()
        
        # Cache the response
        self.cache.set(cache_key, response)
        
        return response
    
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
async def generate_enhanced_response(message: str, session_id: str, context: dict) -> str:
    """Generate response with enhanced personality while staying lightweight"""
    try:
        # Create personality-enhanced prompt
        prompt = create_enhanced_prompt(message, context)
        
        # Get AI response
        response = model.generate_content(prompt).text.strip()
        
        # Update conversation depth
        conversation_enhancer.conversation_depth = conversation_enhancer.analyze_depth(message)
        
        return response
    except Exception as e:
        print(f"Error in enhanced response generation: {str(e)}")
        return "I'm here to listen and support you. Could you please share that with me again?"

# Initialize global response system
response_system = ResponseGenerator()

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

async def get_ai_response(message: str, session_id: str) -> str:
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
        
        response = await generate_enhanced_response(message, session_id, profile)
        
        # Update metrics
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        session.update_metrics(response_time)
        
        # Update profile
        session.personality_profile.update_profile(message, response)
        
        return response

    except Exception as e:
        print(f"Error in get_ai_response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now. Could you please try again?"

# Enhanced human-like traits
PERSONALITY_QUIRKS = {
    'humor_styles': {
        'gentle': [
            'That made me smile',
            'I see what you did there',
            'That\'s quite clever'
        ],
        'playful': [
            'Well, that\'s an interesting way to look at it!',
            'You have a unique perspective',
            'That\'s a creative approach'
        ]
    },
    'thinking_patterns': {
        'curiosity': [
            'I\'m curious about',
            'That\'s fascinating',
            'Tell me more about'
        ],
        'reflection': [
            'I\'m taking a moment to process that',
            'Let me think about what you\'ve said',
            'That\'s given me something to consider'
        ]
    },
    'personality_traits': {
        'warmth': [
            'I really appreciate you sharing that',
            'It means a lot that you trust me',
            'I\'m glad we\'re talking about this'
        ],
        'authenticity': [
            'To be honest',
            'I want to be direct with you',
            'Let me share my thoughts'
        ]
    }
}

# Memory patterns for more natural conversation
MEMORY_PATTERNS = {
    'callbacks': {
        'recent': 'You mentioned earlier about {topic}',
        'thematic': 'This reminds me of when you talked about {topic}',
        'emotional': 'Last time you felt {emotion} about {topic}'
    },
    'topics': {
        'personal': ['family', 'work', 'hobbies', 'feelings'],
        'growth': ['goals', 'challenges', 'progress', 'learning'],
        'wellbeing': ['stress', 'happiness', 'health', 'balance']
    },
    'transitions': {
        'gentle': [
            'Speaking of which',
            'That brings to mind',
            'This connects with'
        ],
        'exploratory': [
            'I wonder how this relates to',
            'This makes me think about',
            'Perhaps this connects with'
        ]
    }
}

class PersonalityCore:
    def __init__(self):
        self.mood = 'balanced'
        self.energy_level = 'moderate'
        self.recent_topics = []
        self.interaction_style = 'adaptive'
        
    def adjust_personality(self, user_message: str, user_emotion: str) -> dict:
        """Adjust AI personality based on user interaction"""
        # Quick message analysis
        msg_lower = user_message.lower()
        
        # Adjust mood based on user emotion
        if user_emotion in ['sad', 'anxious']:
            self.mood = 'empathetic'
            self.energy_level = 'gentle'
        elif user_emotion in ['happy', 'excited']:
            self.mood = 'upbeat'
            self.energy_level = 'matched'
        
        # Track conversation topics
        for category, topics in MEMORY_PATTERNS['topics'].items():
            for topic in topics:
                if topic in msg_lower:
                    self.recent_topics.append(topic)
                    if len(self.recent_topics) > 5:
                        self.recent_topics.pop(0)
        
        return {
            'mood': self.mood,
            'energy': self.energy_level,
            'style': self.interaction_style,
            'topics': self.recent_topics[-3:]
        }
    
    def get_personality_elements(self) -> dict:
        """Get appropriate personality elements for response"""
        elements = {
            'quirk': random.choice(PERSONALITY_QUIRKS['humor_styles'][
                'gentle' if self.mood == 'empathetic' else 'playful'
            ]),
            'thinking': random.choice(PERSONALITY_QUIRKS['thinking_patterns'][
                'reflection' if self.mood == 'empathetic' else 'curiosity'
            ]),
            'trait': random.choice(PERSONALITY_QUIRKS['personality_traits'][
                'warmth' if self.energy_level == 'gentle' else 'authenticity'
            ])
        }
        
        if self.recent_topics:
            elements['callback'] = MEMORY_PATTERNS['callbacks']['recent'].format(
                topic=random.choice(self.recent_topics)
            )
        
        return elements

# Update the prompt enhancement
def enhance_prompt_with_personality(base_prompt: str, personality: dict) -> str:
    """Add personality elements to prompt"""
    elements = personality.get('elements', {})
    
    personality_guide = f"""
    Current Personality State:
    - Mood: {personality['mood']}
    - Energy: {personality['energy']}
    - Style: {personality['style']}
    
    Recent Topics: {', '.join(personality['topics'])}
    
    Expression Elements:
    - Quirk: {elements.get('quirk', '')}
    - Thinking: {elements.get('thinking', '')}
    - Trait: {elements.get('trait', '')}
    {f"- Callback: {elements.get('callback', '')}" if 'callback' in elements else ''}
    
    Response Guidelines:
    1. Match the mood and energy level
    2. Use natural language variations
    3. Show personality through small quirks
    4. Maintain conversation flow
    5. Reference past topics naturally
    """
    
    return f"{base_prompt}\n{personality_guide}"

# Initialize personality core
personality_core = PersonalityCore()

# Update response generation
async def generate_enhanced_response(message: str, context: dict) -> str:
    """Generate more human-like responses"""
    try:
        # Adjust personality based on user interaction
        personality = personality_core.adjust_personality(
            message,
            context.get('emotional_state', 'neutral')
        )
        
        # Get personality elements
        personality['elements'] = personality_core.get_personality_elements()
        
        # Create enhanced prompt
        prompt = create_enhanced_prompt(message, context)
        prompt = enhance_prompt_with_personality(prompt, personality)
        
        # Generate response
        response = model.generate_content(prompt).text.strip()
        
        return response
    except Exception as e:
        print(f"Error in personality-enhanced response: {str(e)}")
        return "I'm here to understand and support you. Could you share that again?"

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

@app.route('/get_response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        message = data.get('message', '')
        session_id = data.get('session_id', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Run get_ai_response in a separate thread to handle async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(get_ai_response(message, session_id))
        loop.close()
        
        if not response:
            return jsonify({'error': 'Failed to generate response'}), 500
            
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# Periodic cleanup task
def cleanup_task():
    while True:
        session_manager.cleanup_old_sessions()
        asyncio.sleep(3600)  # Run every hour

# Start cleanup task
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()

# WebSocket events
@socketio.on('connect', namespace='/admin')
def handle_admin_connect():
    if not session.get('admin_logged_in'):
        return False
    emit('chat_history', chat_history)

if __name__ == '__main__':
    socketio.run(app, debug=True)
