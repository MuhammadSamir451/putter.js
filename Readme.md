# 🏥 Rawalpindi Hospital Chatbot - Complete RAG Implementation

## 📌 Overview

The Rawalpindi Hospital Chatbot is a production-ready, Retrieval Augmented Generation (RAG) system that answers natural language questions about hospitals, departments, and specialists in Rawalpindi, Pakistan. Users can ask questions like "I have heart problems" or "List all hospitals" and receive accurate, data-driven responses. The entire system runs on zero API costs using Puter.js for free Grok AI access. No API keys, no credit card, no billing dashboard. Just pure RAG with semantic search, conversation memory, and complete data retrieval.

## ✨ Features

Natural Language Understanding - "I have problem with urination" automatically finds urologists. Complete Data Retrieval - Shows ALL departments and specialists, never loses data. Conversation Memory - Remembers previous questions for follow-ups like "Tell me about the first one". No Hallucination - Answers ONLY from JSON data, says "I don't know" when information is missing. Symptom-Based Search - "Heart problems" maps to cardiology, "Eye treatment" maps to ophthalmology. Contact Information - Provides phone numbers and locations for each hospital. Session Management - Stores last 20 messages for context awareness. Clear History Button - Reset conversation memory anytime. Quick Action Buttons - One-click access to common queries. Responsive UI - Works on desktop and mobile devices.

## 🛠️ Components and Tech Stack

Backend: Flask handles API routes and serves the frontend. FAISS (Facebook's vector database) performs fast similarity search on embeddings. Sentence-Transformers with the all-MiniLM-L6-v2 model converts text to 384-dimension vectors. Flask-Session manages conversation history storage. Flask-CORS enables frontend-backend communication.

Frontend: HTML5 provides the chat interface structure. CSS3 delivers gradient backgrounds, animations, and responsive design. Vanilla JavaScript handles message display, API calls, and typing indicators. Puter.js provides free Grok API access without API keys.

AI Model: Grok 4.3 from xAI accessed via Puter.js CDN. Temperature set to 0.2 for factual responses. Max tokens set to 1500 per response. Available models include Grok 4.3, Grok 4.1 fast, and Grok 3 - all free for chat. Image generation via Grok 2 costs $0.07 per image.

Data Storage: hospitals.json contains 10 hospitals, 100+ departments, and specialists. FAISS binary index stores 384-dimension embeddings for semantic search. Filesystem session storage maintains user conversation history.

## ⚙️ How It Works

When a user asks a question, the system first retrieves conversation history from the session - the last 6 messages to provide context for follow-up questions. The query is then converted into a vector embedding using Sentence-Transformers, turning text like "I have heart problems" into 384 numbers that represent its meaning. FAISS compares this query embedding against all hospital chunks in the database, calculating distances to find the most semantically similar chunks. The top 15 most relevant results are retrieved. The system then augments the prompt by combining conversation history, retrieved hospital data, and the user's question with strict instructions to answer ONLY from the provided data. This augmented prompt is sent to Puter.js, which forwards it to Grok 4.3. Grok generates a natural, conversational response based solely on the retrieved hospital data, never hallucinating information not present in the JSON. The response is displayed in the chat interface with proper formatting.

For specific query types, the system has optimizations. When users ask "List all hospitals", it bypasses semantic search and directly returns the complete list of all 10 hospitals. When users ask "Show all departments in X hospital", it retrieves the complete department list for that specific hospital. For general queries, semantic search handles everything else.

The session management system stores each user message and bot response with timestamps, keeping only the last 20 messages to manage token limits. This allows follow-up questions like "Tell me about the first hospital" or "What about cardiology there?" to work naturally.

## 📁 Installation

Clone the repository, install dependencies with pip install -r requirements.txt which includes Flask, flask-cors, flask-session, sentence-transformers, faiss-cpu, and numpy. Place your hospitals.json file in the same directory. Run python app.py and open http://localhost:5000 in your browser.

## 📄 License

MIT License - free for personal and commercial use.

Built with ❤️ for the healthcare community | Zero API Costs | True RAG