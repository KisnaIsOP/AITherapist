# Nirya AI Therapist 

A compassionate AI-powered mental health companion that provides emotional support and guidance.

## Live Demo
Visit [Nirya AI Therapist](https://aitherapist.onrender.com) to try it out!

## Features

### Core Features
- Advanced AI-powered conversations using Google's Gemini AI
- Natural and empathetic responses
- Personalized mental health guidance
- Goal-setting and progress tracking
- Emotion recognition and support
- Real-time session analytics

### Admin Dashboard
- Live session tracking
- Real-time analytics
- Interactive charts and visualizations
- User engagement metrics
- Emotion trend analysis
- Mobile-responsive design

## Technical Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, JavaScript, TailwindCSS
- **AI Model**: Google Gemini AI
- **Real-time Updates**: Socket.IO
- **Charts**: Chart.js
- **Deployment**: Render

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/KisnaIsOP/AITherapist.git
cd AITherapist
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with required environment variables:
```env
GOOGLE_API_KEY=your_gemini_api_key
FLASK_SECRET_KEY=your_secret_key
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=your_password_hash
```

5. Run the development server:
```bash
python app.py
```

## Admin Access
- URL: `/admin/login`
- Default credentials:
  - Username: admin
  - Password: new_secure_admin_password_2024!

## Security Features
- Secure session management
- Password hashing
- Protected admin routes
- No persistent user data storage
- Real-time monitoring

## Contributing
Feel free to open issues or submit pull requests. All contributions are welcome!

## License
MIT License

## Author
[Krishna](https://github.com/KisnaIsOP)

## Acknowledgments
- Google Gemini AI for powering the conversational intelligence
- The open-source community for amazing tools and libraries
