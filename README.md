# Nirya - Your AI Mental Health Companion

Nirya is an empathetic AI-powered mental health companion built with Flask and Google's Gemini AI. Named after the Sanskrit word for wisdom and guidance, Nirya provides a supportive space for emotional well-being and personal growth.

## Features

- 🤖 Empathetic AI conversations with personalized support
- 📝 Personal journaling for self-reflection
- 🙏 Gratitude practice tracking
- ⏲️ Mindfulness timer for meditation
- 💭 Emotion tracking and analysis
- ⭐ User feedback system
- 🔗 Mental health resources integration

## Tech Stack

- Backend: Flask (Python)
- AI Model: Google's Gemini AI
- Frontend: HTML, CSS, JavaScript
- Database: SQLAlchemy

## Deployment on Render

### Prerequisites
- Render account
- GitHub repository with your Nirya project
- Google Generative AI API key

### Steps to Deploy

1. **Create a Web Service on Render**
   - Go to Render Dashboard
   - Click "New Web Service"
   - Connect your GitHub repository

2. **Configure Environment Variables**
   Set the following environment variables in Render:
   - `FLASK_SECRET_KEY`: A secure random string
   - `GOOGLE_API_KEY`: Your Google Generative AI API key
   - `ADMIN_USERNAME`: Admin login username
   - `ADMIN_PASSWORD_HASH`: SHA-256 hash of admin password
   - `DATABASE_URL`: Render PostgreSQL database URL

3. **Database Setup**
   - Create a PostgreSQL database on Render
   - Use the provided database URL in `DATABASE_URL`

4. **Deployment Configuration**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

### Local Development

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Set up `.env` file with required variables
5. Run the application: `flask run`

## Security and Privacy

- No personally identifiable information is stored
- Conversations are anonymized
- Admin dashboard requires authentication

## Contributing

Contributions are welcome! Please read our contribution guidelines before submitting a pull request.

## License

This project is licensed under the MIT License.

## Disclaimer

Nirya is an AI companion and not a substitute for professional mental health treatment. Always seek help from a qualified healthcare professional for serious mental health concerns.
