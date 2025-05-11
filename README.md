# Mental Health Support with Medicine Reminder

A comprehensive mental health support application with medicine reminder functionality. This application helps users track their medications, set reminders, monitor their mood, journal their thoughts, take mental health assessments, and receive AI-powered support.

## Features

- **User Authentication**: Secure user registration and login
- **Medication Management**: Track medications, dosages, and schedules
- **Reminder System**: Get notified when it's time to take medications
- **Mood Tracking**: Monitor mood patterns over time, track mood ratings, and view statistics
- **Journal**: Record thoughts and feelings with searchable entries and tagging
- **Mental Health Assessments**: Take standardized assessments like PHQ-9 and GAD-7
- **AI-powered Support**: Get personalized coping strategies and mental health support
- **Resources**: Access educational content and helpful resources

## AI-Powered Mental Health Support

The application includes an advanced AI-powered chatbot that provides personalized mental health support:

### Features

- **Conversational Interface**: Natural, empathetic conversations with users
- **Context-Aware Responses**: Personalized support based on user's mood, medications, journal entries, and assessment results
- **Evidence-Based Guidance**: Responses informed by trusted mental health resources using RAG (Retrieval-Augmented Generation)
- **Crisis Detection**: Automatic detection of crisis situations with appropriate support resources
- **Weekly Reports**: AI-generated summaries of progress and personalized recommendations

### RAG Pipeline

The chatbot uses a Retrieval-Augmented Generation (RAG) pipeline to provide evidence-based responses:

1. **Document Processing**: PDF documents from trusted mental health sources are loaded and processed
2. **Text Chunking**: Documents are split into manageable chunks
3. **Embedding Generation**: Text chunks are converted into vector embeddings
4. **Vector Storage**: Embeddings are stored in a FAISS vector database
5. **Similarity Search**: When a user asks a question, the system retrieves the most relevant information
6. **Response Generation**: The LLM generates a response based on the retrieved information and user context

### Context-Aware Personalization

The chatbot personalizes responses based on:

- Recent mood entries and mood statistics
- Current medications and upcoming reminders
- Journal entries and patterns
- Chat history for conversational context

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: AWS DynamoDB
- **Storage**: AWS S3
- **AI/ML**:
  - Google Gemini Pro for LLM capabilities
  - LangChain for RAG pipeline and orchestration
  - FAISS for vector storage and similarity search
  - PyPDF for document processing

## Documentation

- [Postman Guide](POSTMAN_GUIDE.md): Detailed instructions for testing the API with Postman
- [RAG Implementation](RAG_IMPLEMENTATION.md): Technical details about the RAG system

## Getting Started

### Prerequisites

- Python 3.12+
- AWS Account (for DynamoDB and S3)
- Google API Key (for Gemini Pro)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical_support_reminder.git
   cd medical_support_reminder
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your actual credentials.

5. Create DynamoDB tables:
   ```bash
   python scripts/create_tables.py
   ```

### Running the Application

Start the FastAPI server:
```bash
python run.py
```

Or directly with uvicorn:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001
```

The API will be available at http://localhost:8001, and the API documentation at http://localhost:8001/api/docs.

### Docker

You can also run the application using Docker:

```bash
docker build -t mental-health-app .
docker run -p 8001:8001 --env-file .env mental-health-app
```

## API Endpoints

### Authentication
- `POST /api/auth/register`: Register a new user
- `POST /api/auth/login`: Login and get JWT token
- `GET /api/auth/me`: Get current user profile
- `PUT /api/auth/me`: Update user profile
- `PUT /api/auth/password`: Change password

### Medications
- `GET /api/medications`: List all medications
- `POST /api/medications`: Add a new medication
- `GET /api/medications/{id}`: Get medication details
- `PUT /api/medications/{id}`: Update medication
- `DELETE /api/medications/{id}`: Delete medication
- `POST /api/medications/{id}/image`: Upload medication image

### Reminders
- `GET /api/reminders`: List all reminders
- `GET /api/reminders/today`: Get today's reminders
- `GET /api/reminders/upcoming`: Get upcoming reminders
- `POST /api/reminders`: Create a new reminder
- `PUT /api/reminders/{id}`: Update reminder
- `PUT /api/reminders/{id}/status`: Update reminder status
- `DELETE /api/reminders/{id}`: Delete reminder

### Mood Tracking
- `GET /api/moods`: List all mood entries
- `POST /api/moods`: Create a new mood entry
- `GET /api/moods/{id}`: Get mood entry details
- `PUT /api/moods/{id}`: Update mood entry
- `DELETE /api/moods/{id}`: Delete mood entry
- `GET /api/moods/stats`: Get mood statistics

### Journal
- `GET /api/journal`: List all journal entries
- `POST /api/journal`: Create a new journal entry
- `GET /api/journal/{id}`: Get journal entry details
- `PUT /api/journal/{id}`: Update journal entry
- `DELETE /api/journal/{id}`: Delete journal entry
- `GET /api/journal/search`: Search journal entries by content or tags

### AI Support
- `POST /api/ai/chat`: Send a message to the chatbot
- `GET /api/ai/chat/history`: Get chat history
- `GET /api/ai/recommendations`: Get personalized recommendations
- `GET /api/ai/weekly-report`: Get weekly progress report

## License

This project is licensed under the MIT License - see the LICENSE file for details.
