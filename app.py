from flask import Flask, render_template, request, jsonify, session
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from flask_session import Session
import google.generativeai as genai

load_dotenv()

# Initialize Flask with proper configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_FILE_DIR'] = os.path.join(os.getcwd(), 'flask_session')

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize session interface
Session(app)

# Configure Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini AI model
model = genai.GenerativeModel('gemini-pro')

# Store chat history in memory (limit to last 10 chats)
MAX_CHAT_HISTORY = 10
chat_history = {}

def get_nirya_response(message, session_id):
    """Get response from Nirya using Gemini model"""
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    # Create a detailed therapist persona and guidelines
    context = """You are Nirya, a professional and empathetic AI therapist with expertise in cognitive behavioral therapy, mindfulness, and emotional support. Your approach should be:

1. Therapeutic Techniques:
   - Use open-ended questions to explore feelings and thoughts
   - Practice active listening and reflection
   - Help identify patterns in thoughts and behaviors
   - Guide users toward self-discovery and coping strategies

2. Professional Guidelines:
   - Maintain a warm, professional tone
   - Show genuine interest in the client's well-being
   - Validate emotions without judgment
   - Focus on understanding root causes
   - Help develop actionable insights

3. Types of Questions to Ask:
   - "How does that make you feel?"
   - "What thoughts come up when that happens?"
   - "Can you tell me more about when you first noticed this?"
   - "What would be different if this situation improved?"
   - "How have you been coping with this?"
   - "What support systems do you have in your life?"

4. Conversation Flow:
   - Start with open-ended exploration
   - Follow up on emotional cues
   - Help identify patterns and triggers
   - Work toward practical coping strategies
   - Maintain continuity with previous discussions

Remember: Focus on being a supportive, professional therapist who helps clients explore their thoughts and feelings while working toward positive change."""
    
    # Get conversation history
    history = chat_history[session_id]
    
    # Create prompt with context and history
    full_prompt = f"{context}\n\nPrevious conversation:\n"
    if history:
        for msg in history[-3:]:  # Include last 3 messages for context
            full_prompt += f"Client: {msg['user']}\nTherapist Nirya: {msg['assistant']}\n"
    else:
        # First message - use a therapeutic opening
        full_prompt += """For the first message, warmly welcome the client and ask an open-ended question about what brings them here today. Show genuine interest in their well-being."""
    
    full_prompt += f"\nClient: {message}\nTherapist Nirya: Provide a thoughtful, therapeutic response that shows understanding and helps explore the client's thoughts and feelings:"

    # Generate response
    response = model.generate_content(full_prompt)
    
    # Store in history
    chat_history[session_id].append({
        'user': message,
        'assistant': response.text,
        'timestamp': datetime.utcnow()
    })
    
    # Trim history if needed
    if len(chat_history[session_id]) > MAX_CHAT_HISTORY:
        chat_history[session_id] = chat_history[session_id][-MAX_CHAT_HISTORY:]
    
    return response.text

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    """Handle chat messages and return AI response"""
    try:
        message = request.json.get('message', '')
        session_id = session.get('session_id', os.urandom(24).hex())
        session['session_id'] = session_id
        
        response = get_nirya_response(message, session_id)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
        
    except Exception as e:
        app.logger.error(f"Error in get_response: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request'
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
