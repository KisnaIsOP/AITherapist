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
    
    # Create a more detailed context for Nirya
    context = """You are Nirya, an empathetic AI therapist. Your responses should be:
    - Natural and varied (never repeat the same phrase)
    - Thoughtful and engaging
    - Focused on understanding and helping the user
    - Encouraging open dialogue
    
    Guidelines:
    - Ask specific questions based on what the user shares
    - Show genuine interest in their thoughts and feelings
    - Avoid generic responses like "I'm here to support you"
    - Each response should be unique and personalized
    - If the user seems unsure what to talk about, suggest specific topics or ask about their day
    """
    
    # Get conversation history
    history = chat_history[session_id]
    
    # Create prompt with context and history
    full_prompt = f"{context}\n\nPrevious conversation:\n"
    if history:
        for msg in history[-3:]:  # Include last 3 messages for context
            full_prompt += f"User: {msg['user']}\nNirya: {msg['assistant']}\n"
    
    full_prompt += f"\nUser: {message}\nNirya: Please provide a thoughtful, non-repetitive response that moves the conversation forward:"

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
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
