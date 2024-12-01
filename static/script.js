document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const feedbackContainer = document.querySelector('.feedback-container');
    const feedbackStars = document.querySelectorAll('.feedback-stars i');
    const submitFeedback = document.getElementById('submit-feedback');
    
    let currentRating = 0;
    let conversationCount = 0;
    let sessionId = localStorage.getItem('session_id') || Date.now().toString();
    
    // Store session ID
    localStorage.setItem('session_id', sessionId);

    // Initialize engagement features
    const journalSection = document.querySelector('.journal-section');
    const gratitudeSection = document.querySelector('.gratitude-section');
    const mindfulnessTimer = document.querySelector('.mindfulness-timer');
    
    // Message handling
    function addMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'therapist'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const text = document.createElement('p');
        text.textContent = content;
        
        const timestamp = document.createElement('span');
        timestamp.className = 'timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageContent.appendChild(text);
        messageContent.appendChild(timestamp);
        messageDiv.appendChild(messageContent);
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Increment conversation count and show feedback after every 5 messages
        conversationCount++;
        if (conversationCount % 5 === 0) {
            showFeedback();
        }
    }

    // Send message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Disable input and button while sending
        userInput.disabled = true;
        sendButton.disabled = true;

        addMessage(message, true);
        userInput.value = '';
        userInput.style.height = 'auto';

        try {
            const response = await fetch('/get_response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    session_id: sessionId
                })
            });

            const data = await response.json();
            
            if (data.error) {
                addMessage("I apologize, but I'm having trouble responding right now. Please try again.", false);
            } else {
                addMessage(data.response, false);

                // Check if response contains engagement prompts
                if (data.response.includes('journal') || data.response.includes('write down')) {
                    showJournalSection();
                } else if (data.response.includes('grateful') || data.response.includes('thankful')) {
                    showGratitudeSection();
                } else if (data.response.includes('breathing') || data.response.includes('mindful')) {
                    showMindfulnessTimer();
                }
            }

        } catch (error) {
            console.error('Error:', error);
            addMessage("I apologize, but I'm having trouble responding right now. Please try again.", false);
        } finally {
            // Re-enable input and button
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    // Event listeners for sending messages
    sendButton.addEventListener('click', function(e) {
        e.preventDefault();
        sendMessage();
    });

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Feedback handling
    function showFeedback() {
        feedbackContainer.style.display = 'block';
    }

    feedbackStars.forEach(star => {
        star.addEventListener('mouseover', function() {
            const rating = this.dataset.rating;
            updateStars(rating);
        });

        star.addEventListener('mouseout', function() {
            updateStars(currentRating);
        });

        star.addEventListener('click', function() {
            currentRating = this.dataset.rating;
            updateStars(currentRating);
        });
    });

    function updateStars(rating) {
        feedbackStars.forEach(star => {
            const starRating = star.dataset.rating;
            star.className = starRating <= rating ? 'fas fa-star' : 'far fa-star';
        });
    }

    submitFeedback.addEventListener('click', async function() {
        const feedbackText = document.getElementById('feedback-text').value;
        try {
            await fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    score: currentRating,
                    feedback: feedbackText
                })
            });
            feedbackContainer.style.display = 'none';
            currentRating = 0;
            updateStars(0);
            document.getElementById('feedback-text').value = '';
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    });

    // Engagement feature handlers
    function showJournalSection() {
        journalSection.style.display = 'block';
        gratitudeSection.style.display = 'none';
        mindfulnessTimer.style.display = 'none';
    }

    function showGratitudeSection() {
        gratitudeSection.style.display = 'block';
        journalSection.style.display = 'none';
        mindfulnessTimer.style.display = 'none';
    }

    function showMindfulnessTimer() {
        mindfulnessTimer.style.display = 'block';
        journalSection.style.display = 'none';
        gratitudeSection.style.display = 'none';
    }

    // Journal handling
    document.getElementById('save-journal').addEventListener('click', function() {
        const entry = document.getElementById('journal-entry').value;
        if (entry) {
            document.getElementById('journal-entry').value = '';
            journalSection.style.display = 'none';
            addMessage("Thank you for sharing your thoughts. Would you like to explore these feelings further?", false);
        }
    });

    // Gratitude list handling
    document.getElementById('add-gratitude').addEventListener('click', function() {
        const gratitudeInput = document.getElementById('gratitude-input');
        const gratitudeList = document.getElementById('gratitude-list');
        
        if (gratitudeInput.value) {
            const li = document.createElement('li');
            li.textContent = gratitudeInput.value;
            gratitudeList.appendChild(li);
            gratitudeInput.value = '';
            
            if (gratitudeList.children.length >= 3) {
                gratitudeSection.style.display = 'none';
                addMessage("It's wonderful to practice gratitude. How do these positive reflections make you feel?", false);
            }
        }
    });

    // Mindfulness timer handling
    let timerInterval;
    document.getElementById('start-timer').addEventListener('click', function() {
        const timerDisplay = document.querySelector('.timer-display');
        let timeLeft = 300; // 5 minutes in seconds
        
        this.disabled = true;
        
        timerInterval = setInterval(() => {
            timeLeft--;
            const minutes = Math.floor(timeLeft / 60);
            const seconds = timeLeft % 60;
            timerDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            if (timeLeft <= 0) {
                clearInterval(timerInterval);
                this.disabled = false;
                mindfulnessTimer.style.display = 'none';
                addMessage("How do you feel after taking this mindful moment?", false);
            }
        }, 1000);
    });
});
