from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv
import random

app = Flask(__name__)
app.secret_key = os.urandom(24)
load_dotenv()

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
        user_session_data = session['user_session']
        user_session = UserSession()
        user_session.metrics.conversation_length = user_session_data['metrics']['conversation_length']
        user_session.metrics.emotional_patterns = user_session_data['metrics']['emotional_patterns']
        user_session.metrics.feedback_scores = user_session_data['metrics']['feedback_scores']
        user_session.metrics.recurring_themes = set(user_session_data['metrics']['recurring_themes'])
        user_session.metrics.session_duration = user_session_data['metrics']['session_duration']
        user_session.metrics.helpful_responses = user_session_data['metrics']['helpful_responses']
        user_session.current_emotions = user_session_data['current_emotions']
        user_session.themes_discussed = set(user_session_data['themes_discussed'])
        user_session.last_topics = user_session_data['last_topics']
        user_session.gratitude_entries = user_session_data['gratitude_entries']
        user_session.journal_entries = user_session_data['journal_entries']

        # Analyze emotions and update metrics
        current_emotions = analyze_emotion(user_input)
        user_session.current_emotions = current_emotions
        user_session.metrics.conversation_length += 1

        # Track emotional patterns
        for emotion in current_emotions:
            user_session.metrics.emotional_patterns[emotion] = user_session.metrics.emotional_patterns.get(emotion, 0) + 1

        # Determine if we should offer a micro-challenge
        should_offer_challenge = user_session.metrics.conversation_length % 3 == 0
        micro_challenge = suggest_micro_challenge(current_emotions, user_session.themes_discussed) if should_offer_challenge else None

        # Enhanced prompt with emotional awareness and context
        prompt = f"{THERAPEUTIC_CONTEXT}\n\n"
        prompt += f"Current emotional state: {', '.join(current_emotions) if current_emotions else 'neutral'}\n"
        prompt += f"Conversation history:\n{conversation_history}\n\n"
        prompt += f"User: {user_input}\n\n"
        
        if micro_challenge:
            prompt += f"Consider suggesting this micro-challenge: {micro_challenge}\n\n"
        
        prompt += "Respond with empathy, using the following approach:\n"
        prompt += "1. Acknowledge and validate their feelings\n"
        prompt += "2. Paraphrase their message to show understanding\n"
        prompt += "3. Ask a relevant follow-up question\n"
        prompt += "4. If appropriate, suggest the micro-challenge\n\n"
        prompt += "Therapist:"

        response = model.generate_content(prompt)
        
        # Update session metrics
        if any(phrase in response.text.lower() for phrase in ['thank you', 'helps', 'helpful']):
            user_session.metrics.helpful_responses += 1
        
        session['user_session'] = user_session.to_dict()
        return response.text

    except Exception as e:
        return "I want to make sure I understand what you're sharing. Could you tell me more about that?"

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

    # Get conversation history
    conversation_history = session.get('conversation_history', [])
    history_text = "\n".join([f"{'User' if i%2==0 else 'Therapist'}: {msg}" 
                             for i, msg in enumerate(conversation_history)])

    # Get AI response
    ai_response = get_ai_response(user_message, history_text)

    # Update conversation history
    conversation_history.extend([user_message, ai_response])
    session['conversation_history'] = conversation_history[-10:]  # Keep last 10 messages

    # Get session metrics
    user_session = session.get('user_session')
    metrics = user_session['metrics'] if user_session else {}

    return jsonify({
        'response': ai_response,
        'timestamp': datetime.now().strftime("%H:%M"),
        'metrics': metrics
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint for collecting user feedback"""
    feedback_data = request.json
    if 'user_session' in session:
        user_session_data = session['user_session']
        user_session = UserSession()
        user_session.metrics.feedback_scores = user_session_data['metrics']['feedback_scores']
        user_session.metrics.feedback_scores.append(feedback_data.get('score'))
        session['user_session'] = user_session.to_dict()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
