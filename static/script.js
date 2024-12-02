document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const feedbackContainer = document.querySelector('.feedback-container');
    const feedbackStars = document.querySelectorAll('.feedback-stars i');
    const submitFeedback = document.getElementById('submit-feedback');
    const themeToggle = document.getElementById('themeToggle');
    
    let currentRating = 0;
    let conversationCount = 0;
    let sessionId = localStorage.getItem('session_id') || Date.now().toString();
    
    // Store session ID
    localStorage.setItem('session_id', sessionId);

    // Initialize engagement features
    const journalSection = document.querySelector('.journal-section');
    const gratitudeSection = document.querySelector('.gratitude-section');
    const mindfulnessTimer = document.querySelector('.mindfulness-timer');
    
    // Theme Management
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (prefersDarkScheme.matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    // Theme toggle handler
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        // Add animation class
        themeToggle.classList.add('theme-transition');
        setTimeout(() => themeToggle.classList.remove('theme-transition'), 300);
    });

    // Enhanced Emoji Mapping with more contextual categories
    const emojiMap = {
        greeting: ['👋', '🤗', '😊', '✨', '💫'],
        farewell: ['👋', '💫', '🌟', '✨', '🤗'],
        positive: ['😊', '🌟', '💫', '✨', '🎉', '💖', '🌈'],
        concern: ['😔', '💙', '🫂', '💭', '🤍'],
        empathy: ['💗', '🫂', '💝', '💖', '🤗'],
        reflection: ['🤔', '💭', '💡', '✨', '🌟'],
        listening: ['👂', '🎧', '💭', '🤝', '💫'],
        support: ['💪', '🤗', '🌈', '💫', '💝'],
        gratitude: ['🙏', '💖', '✨', '💫', '🌟'],
        encouragement: ['💫', '🌟', '✨', '💪', '🔆'],
        understanding: ['💭', '🤝', '💡', '💫', '💖'],
        sympathy: ['💙', '🫂', '🤗', '💖', '💝'],
        celebration: ['🎉', '🌟', '✨', '🎊', '💫'],
        mindfulness: ['🧘‍♀️', '🌱', '🍃', '🌸', '💫'],
        growth: ['🌱', '🚀', '📈', '💫', '⭐'],
        healing: ['💖', '🌈', '✨', '🌸', '💫'],
        strength: ['💪', '🦁', '⭐', '🌟', '💫'],
        peace: ['🕊️', '☮️', '🌸', '✨', '💫'],
        anxiety: ['😮‍💨', '💭', '🫂', '💙', '🤍'],
        depression: ['💙', '🫂', '💗', '🤍', '💝'],
        hope: ['🌅', '🌈', '✨', '💫', '🌟'],
        sleep: ['😴', '💤', '🌙', '✨', '🌟'],
        stress: ['😮‍💨', '🫂', '💆‍♀️', '🌸', '💫'],
        confidence: ['💪', '👑', '⭐', '🌟', '💫']
    };

    // Enhanced context detection for better emoji placement
    function addEmojisToResponse(response) {
        // Check if response is undefined, null, or not a string
        if (!response || typeof response !== 'string') {
            return response || ''; // Return empty string if response is invalid
        }

        const lowerResponse = response.toLowerCase();
        let enhancedResponse = response;
        
        // Greeting detection
        if (/^(hi|hello|hey|greetings|good (morning|afternoon|evening))/i.test(response)) {
            enhancedResponse = getRandomEmoji('greeting') + ' ' + enhancedResponse;
        }
        
        // Farewell detection
        if (/(goodbye|bye|take care|see you|until next time)/i.test(lowerResponse)) {
            enhancedResponse += ' ' + getRandomEmoji('farewell');
        }
        
        // Emotional support detection
        if (/(here for you|support you|understand|must be hard|difficult time)/i.test(lowerResponse)) {
            enhancedResponse = getRandomEmoji('empathy') + ' ' + enhancedResponse;
        }
        
        // Encouragement detection
        if (/(you can do|believe in|keep going|step forward|progress)/i.test(lowerResponse)) {
            enhancedResponse += ' ' + getRandomEmoji('encouragement');
        }
        
        // Anxiety/Stress detection
        if (/(anxious|worried|stress|overwhelm|nervous)/i.test(lowerResponse)) {
            enhancedResponse = getRandomEmoji('anxiety') + ' ' + enhancedResponse;
        }
        
        // Depression/Sadness detection
        if (/(sad|depress|down|lonely|hopeless)/i.test(lowerResponse)) {
            enhancedResponse = getRandomEmoji('depression') + ' ' + enhancedResponse;
        }
        
        // Hope/Positivity detection
        if (/(hope|better days|future|positive|bright)/i.test(lowerResponse)) {
            enhancedResponse += ' ' + getRandomEmoji('hope');
        }
        
        // Mindfulness/Relaxation detection
        if (/(breathe|relax|mindful|present|calm)/i.test(lowerResponse)) {
            enhancedResponse = getRandomEmoji('mindfulness') + ' ' + enhancedResponse;
        }
        
        // Gratitude detection
        if (/(thank|grateful|appreciate|blessed)/i.test(lowerResponse)) {
            enhancedResponse += ' ' + getRandomEmoji('gratitude');
        }
        
        // Sleep/Rest detection
        if (/(sleep|rest|tired|exhausted|night)/i.test(lowerResponse)) {
            enhancedResponse = getRandomEmoji('sleep') + ' ' + enhancedResponse;
        }
        
        // Confidence/Strength detection
        if (/(strong|capable|achieve|proud|confidence)/i.test(lowerResponse)) {
            enhancedResponse += ' ' + getRandomEmoji('confidence');
        }
        
        return enhancedResponse;
    }

    // Get random emoji with weighted selection
    function getRandomEmoji(category) {
        const emojis = emojiMap[category];
        if (!emojis) return '';
        
        // Add variation to prevent repetitive emojis
        const lastUsedEmoji = sessionStorage.getItem(`last_${category}_emoji`);
        let selectedEmoji;
        
        do {
            selectedEmoji = emojis[Math.floor(Math.random() * emojis.length)];
        } while (selectedEmoji === lastUsedEmoji && emojis.length > 1);
        
        sessionStorage.setItem(`last_${category}_emoji`, selectedEmoji);
        return selectedEmoji;
    }

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

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const aiResponse = data.response || 'I apologize, but I could not generate a response.';
            
            const enhancedResponse = addEmojisToResponse(aiResponse);
            addMessage(enhancedResponse, false);
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error processing your message. Please try again.', false);
        } finally {
            // Re-enable input and button
            userInput.disabled = false;
            sendButton.disabled = false;
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
