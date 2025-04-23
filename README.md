# Music Emotion Recommender

This project analyzes speech audio to detect emotions and provides music recommendations based on the detected emotional state.

## Setup Instructions

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Environment Variables:
The project uses environment variables for all paths and API keys. A `.env` file has been created with the necessary configurations. You may need to adjust the paths based on your system configuration.

Example `.env` file:
```
# Vector Database Paths
VECTOR_DB_BASE_PATH="/path/to/your/RAG/vector_db"

# VAD Model Paths
VALENCE_CHECKPOINT_PATH="/path/to/your/VAD_Models/V.pth"
AD_CHECKPOINT_PATH="/path/to/your/VAD_Models/AD.pth"

# Embedding Model
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# JSON Files
EMOTION_CONTEXT_PATH="emotion_context.json"

# API Keys (replace with your own keys)
SERPAPI_KEY="your-serp-api-key"

# PubMed Email
PUBMED_EMAIL="your-email@example.com"

# Test Audio Paths (optional)
TEST_AUDIO_PATH="/path/to/test/audio.wav"
```

## Usage

Run the main script:
```
python music_emotion_recommender.py
```

This will:
1. Analyze the provided audio file to detect emotions directly (no transcription needed)
2. Generate music recommendations based on the detected emotions
3. Save the results to a prompt.txt file

## Project Structure

- `music_emotion_recommender.py`: Main script for audio processing and recommendation
- `VAD_Models/`: Contains models for Valence, Arousal, and Dominance prediction
- `RAG/`: Retrieval-Augmented Generation components
- `emotion_context.json`: Mapping of emotions to musical characteristics
- `requirements.txt`: Unified requirements file for all project dependencies

## Technical Details

The system works by:
1. Extracting VAD (Valence, Arousal, Dominance) scores directly from audio
2. Predicting emotions based on the VAD scores
3. Generating music recommendations based on the detected emotions

All paths and API keys are stored in the `.env` file for better security and configuration management. 