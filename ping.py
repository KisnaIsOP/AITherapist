import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

def ping_server():
    """Ping the server every 14 minutes to prevent spin down"""
    url = os.getenv('APP_URL', 'https://nirya-ai-therapist.onrender.com')  # Replace with your Render URL
    
    while True:
        try:
            response = requests.get(url)
            print(f"Ping status: {response.status_code}")
        except Exception as e:
            print(f"Ping failed: {str(e)}")
        
        # Sleep for 14 minutes (840 seconds)
        # Render free tier instances spin down after 15 minutes of inactivity
        time.sleep(840)

if __name__ == "__main__":
    print("Starting ping service...")
    ping_server()
