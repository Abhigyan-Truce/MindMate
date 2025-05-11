# Postman Guide for Mental Health Support API

This guide provides instructions for testing the Mental Health Support API using Postman.

## Setup

1. Download and install [Postman](https://www.postman.com/downloads/)
2. Import the collection (optional):
   - Download the Postman collection JSON file (if available)
   - In Postman, click "Import" and select the downloaded file
3. Set up environment variables:
   - Create a new environment in Postman
   - Add the following variables:
     - `base_url`: `http://localhost:8001`
     - `token`: (leave empty initially)

## Authentication

### Register a New User

- **Method**: POST
- **URL**: `{{base_url}}/api/auth/register`
- **Headers**:
  - Content-Type: application/json
- **Body** (raw JSON):
  ```json
  {
    "email": "user@example.com",
    "name": "Test User",
    "password": "password123"
  }
  ```

### Login

- **Method**: POST
- **URL**: `{{base_url}}/api/auth/login`
- **Headers**:
  - Content-Type: application/json
- **Body** (raw JSON):
  ```json
  {
    "email": "user@example.com",
    "password": "password123"
  }
  ```
- **After successful login**:
  - Copy the `access_token` from the response
  - Set the `token` environment variable to this value

### Get Current User Profile

- **Method**: GET
- **URL**: `{{base_url}}/api/auth/me`
- **Headers**:
  - Authorization: Bearer {{token}}

## Medications

### List All Medications

- **Method**: GET
- **URL**: `{{base_url}}/api/medications`
- **Headers**:
  - Authorization: Bearer {{token}}

### Add a New Medication

- **Method**: POST
- **URL**: `{{base_url}}/api/medications`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "name": "Aspirin",
    "dosage": "100mg",
    "frequency": "daily",
    "time_of_day": "morning",
    "specific_times": ["08:00"],
    "start_date": "2023-09-01",
    "medication_type": "pill"
  }
  ```

### Get Medication Details

- **Method**: GET
- **URL**: `{{base_url}}/api/medications/{medication_id}`
- **Headers**:
  - Authorization: Bearer {{token}}

### Update Medication

- **Method**: PUT
- **URL**: `{{base_url}}/api/medications/{medication_id}`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "name": "Aspirin",
    "dosage": "200mg",
    "frequency": "daily",
    "time_of_day": "morning",
    "specific_times": ["08:00"],
    "start_date": "2023-09-01",
    "medication_type": "pill"
  }
  ```

### Delete Medication

- **Method**: DELETE
- **URL**: `{{base_url}}/api/medications/{medication_id}`
- **Headers**:
  - Authorization: Bearer {{token}}

## Reminders

### List All Reminders

- **Method**: GET
- **URL**: `{{base_url}}/api/reminders`
- **Headers**:
  - Authorization: Bearer {{token}}

### Create a New Reminder

- **Method**: POST
- **URL**: `{{base_url}}/api/reminders`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "medication_id": "{medication_id}",
    "scheduled_time": "2023-09-01T08:00:00",
    "status": "pending"
  }
  ```

### Update Reminder Status

- **Method**: PUT
- **URL**: `{{base_url}}/api/reminders/{reminder_id}/status`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "status": "completed"
  }
  ```

## Mood Tracking

### List All Mood Entries

- **Method**: GET
- **URL**: `{{base_url}}/api/moods`
- **Headers**:
  - Authorization: Bearer {{token}}

### Create a New Mood Entry

- **Method**: POST
- **URL**: `{{base_url}}/api/moods`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "mood_rating": 8,
    "tags": ["happy", "energetic"],
    "notes": "Feeling great today!"
  }
  ```

### Get Mood Statistics

- **Method**: GET
- **URL**: `{{base_url}}/api/moods/stats`
- **Headers**:
  - Authorization: Bearer {{token}}

## Journal

### List All Journal Entries

- **Method**: GET
- **URL**: `{{base_url}}/api/journal`
- **Headers**:
  - Authorization: Bearer {{token}}

### Create a New Journal Entry

- **Method**: POST
- **URL**: `{{base_url}}/api/journal`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "title": "My First Journal Entry",
    "content": "Today was a great day. I felt energetic and accomplished a lot of tasks.",
    "tags": ["productive", "happy"]
  }
  ```

### Search Journal Entries

- **Method**: GET
- **URL**: `{{base_url}}/api/journal/search?query=great`
- **Headers**:
  - Authorization: Bearer {{token}}

## AI Support

### Send a Message to the Chatbot

- **Method**: POST
- **URL**: `{{base_url}}/api/ai/chat`
- **Headers**:
  - Content-Type: application/json
  - Authorization: Bearer {{token}}
- **Body** (raw JSON):
  ```json
  {
    "message": "How can I manage my anxiety?"
  }
  ```

### Get Chat History

- **Method**: GET
- **URL**: `{{base_url}}/api/ai/chat/history`
- **Headers**:
  - Authorization: Bearer {{token}}

### Get Personalized Recommendations

- **Method**: GET
- **URL**: `{{base_url}}/api/ai/recommendations`
- **Headers**:
  - Authorization: Bearer {{token}}

### Get Weekly Progress Report

- **Method**: GET
- **URL**: `{{base_url}}/api/ai/weekly-report`
- **Headers**:
  - Authorization: Bearer {{token}}

## Tips for Testing

1. **Authentication Flow**:
   - Register a new user
   - Login to get the token
   - Use the token for all subsequent requests

2. **Testing Dependencies**:
   - Create a medication before creating a reminder
   - Use valid IDs when referencing other resources

3. **Error Handling**:
   - Test with invalid data to see error responses
   - Test with missing authentication to verify security

4. **AI Features**:
   - Add some mood entries and journal entries before testing AI features
   - Try different types of questions with the chatbot
   - Check if recommendations change based on your mood entries
