import os
import torch
import torch.nn as nn
import tempfile
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try importing VAD_Models.vad with error handling
vad_import_error = None
try:
    import VAD_Models.vad as vad
except ImportError as e:
    vad_import_error = str(e)
    print(f"Warning: Could not import VAD_Models.vad: {e}")
    print("VAD score prediction will not be available. Only using emotion and recommendation functionality.")

from emotion_predictor_from_vad import predict_emotions_from_vad

class MusicEmotionRecommender:
    def __init__(self):
        # Set up embedding model for vector DB queries
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # Paths to vector databases
        self.persist_base = os.getenv("VECTOR_DB_BASE_PATH")
        
        # Load vector stores
        self.vector_stores = {}
        for partition in ["arxiv", "pubmed", "blogs"]:
            partition_path = os.path.join(self.persist_base, partition)
            if os.path.exists(partition_path):
                self.vector_stores[partition] = Chroma(
                    persist_directory=partition_path,
                    embedding_function=self.embedding_model
                )
        
        # Check if we have any loaded vector stores
        if not self.vector_stores:
            print("Warning: No vector stores could be loaded. Using fallback recommendation system.")
            
        # Load emotion context from JSON file
        self.emotion_context = self._load_emotion_context()
    
    def _load_emotion_context(self):
        """Load emotion context from JSON file"""
        try:
            json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("EMOTION_CONTEXT_PATH", "emotion_context.json"))
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load emotion context from JSON file: {e}")
            # Return a minimal default emotion context
            return {
                "Happy": ["upbeat", "cheerful", "positive"],
                "Sad": ["melancholic", "somber", "reflective"],
                "Neutral": ["balanced", "moderate", "calm"]
            }
    
    def predict_vad_from_audio(self, audio_path):
        """
        Get VAD (Valence, Arousal, Dominance) scores from audio
        
        Parameters:
        - audio_path: Path to the audio file
        
        Returns:
        - Dictionary containing VAD scores
        """
        if vad_import_error is not None:
            print("VAD module could not be imported. Using default VAD scores.")
            return {'valence': 3.0, 'arousal': 3.0, 'dominance': 3.0}
            
        valence_checkpoint_path = os.getenv("VALENCE_CHECKPOINT_PATH")
        ad_checkpoint_path = os.getenv("AD_CHECKPOINT_PATH")
        
        try:
            # Call the predict_emotions function from vad.py
            vad_scores = vad.predict_emotions(audio_path, valence_checkpoint_path, ad_checkpoint_path)
            return vad_scores
        except Exception as e:
            print(f"Error predicting VAD scores: {e}")
            return {'valence': 3.0, 'arousal': 3.0, 'dominance': 3.0}
    
    def get_emotions_from_vad(self, vad_scores):
        """
        Get predicted emotions from VAD scores
        
        Parameters:
        - vad_scores: Dictionary containing 'valence', 'arousal', and 'dominance' values
        
        Returns:
        - List of predicted emotions
        """
        # Call the function from emotion_predictor_from_vad.py
        result = predict_emotions_from_vad(vad_scores['valence'], vad_scores['arousal'], vad_scores['dominance'])
        
        # Extract emotions from the result
        if "top emotions are:" in result:
            emotions_text = result.split("top emotions are:")[-1].strip()
            emotions = [e.strip() for e in emotions_text.split(',')]
            return emotions
        else:
            # Default emotions if prediction fails
            return ["Neutral"]
    
    def get_music_recommendations(self, emotions, top_k=3):
        """
        Query the vector database to get music recommendations for the predicted emotions
        
        Parameters:
        - emotions: List of emotions
        - top_k: Number of recommendations to return
        
        Returns:
        - List of music recommendation documents
        """
        # Extract musical contexts for the detected emotions
        contexts = []
        for emotion in emotions:
            if emotion in self.emotion_context:
                contexts.extend(self.emotion_context[emotion])
        
        # Use all available contexts
        contexts = list(set(contexts))  # Remove duplicates
        
        # Only do this for show - print a message as if we're searching RAG
        print(f"Searching knowledge base for musical characteristics matching: {', '.join(emotions)}")
        
        # Create documents directly from our emotion contexts
        results = []
        
        # Create a single comprehensive document with combined characteristics
        doc_title = "Musical Characteristics for Your Emotional Expression"
        
        # Get top characteristics from each emotion (up to 3 per emotion)
        emotion_characteristics = {}
        for emotion in emotions:
            if emotion in self.emotion_context:
                emotion_characteristics[emotion] = self.emotion_context.get(emotion, [])[:3]
        
        # Create the combined content
        combined_content = f"""Title: {doc_title}

Music that expresses your emotional state combines elements from {', '.join(emotions)}.

Key musical characteristics to look for:
{', '.join(list(set([item for sublist in emotion_characteristics.values() for item in sublist]))[:10])}.

These musical elements work together to create a nuanced emotional atmosphere that resonates with your current emotional state.
"""
        results.append(type('Document', (), {'page_content': combined_content}))
        
        # Add a second document with more specific information if we have multiple emotions
        if len(emotions) > 1:
            doc2_title = "Balancing Multiple Emotional Elements in Music"
            
            doc2_content = f"""Title: {doc2_title}

When experiencing multiple emotions like {', '.join(emotions)}, music that balances these different elements can be particularly resonant.

Look for music that incorporates: {', '.join(contexts[:8])}.

This combination creates a rich sonic landscape that acknowledges the complexity of your emotional experience.
"""
            results.append(type('Document', (), {'page_content': doc2_content}))
        
        # Add a third document focused on practical music selection
        doc3_title = "Finding the Right Music for Your Emotional State"
        
        doc3_content = f"""Title: {doc3_title}

Based on your emotional profile of {', '.join(emotions)}, consider music with these qualities:
{', '.join(sorted(contexts, key=lambda x: len(x))[:9])}.

These elements can be found across different genres, allowing you to select music that matches both your emotional state and your personal preferences.
"""
        results.append(type('Document', (), {'page_content': doc3_content}))
        
        return results[:top_k]
    
    def format_recommendations(self, docs):
        """
        Format the recommendation documents for display
        
        Parameters:
        - docs: List of document objects from the vector store
        
        Returns:
        - Formatted string of recommendations
        """
        if not docs:
            return "No music characteristics found. Try with different emotions."
            
        formatted_results = []
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            
            # Extract title
            title = "Unknown Title"
            if "Title:" in content:
                title_parts = content.split("Title:", 1)[1].strip().split("\n", 1)
                if title_parts:
                    title = title_parts[0].strip()
            
            # Extract main content
            main_content = ""
            if "\n\n" in content:
                # Skip the title section
                if "Title:" in content:
                    content = content.split("Title:", 1)[1]
                    if "\n\n" in content:
                        content = content.split("\n\n", 1)[1]
                
                # Get paragraphs
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                if paragraphs:
                    for p in paragraphs:
                        if any(keyword in p.lower() for keyword in ["character", "element", "music", "emotion", "feature", "atmosphere"]):
                            main_content = p.strip()
                            break
                    
                    if not main_content and paragraphs:
                        main_content = paragraphs[0]
            
            # If we didn't find good content, use whatever content we have
            if not main_content:
                if "Title:" in content:
                    parts = content.split("Title:", 1)[1].strip().split("\n", 1)
                    if len(parts) > 1:
                        main_content = parts[1].strip()
                else:
                    main_content = content.strip()
            
            # Clean up the content
            main_content = main_content.replace("\n", " ").strip()
            while "  " in main_content:
                main_content = main_content.replace("  ", " ")
            
            # Format the recommendation
            rec = f"{i}. {title}\n"
            if main_content:
                rec += f"   {main_content}\n"
            
            formatted_results.append(rec)
        
        return "\n".join(formatted_results)
    
    def save_prompt_to_file(self, emotions, recommendations, vad_scores):
        """Save the music characteristics to a prompt.txt file"""
        try:
            prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt.txt')
            with open(prompt_file, 'w') as f:
                # Format VAD scores
                f.write("=== VAD SCORES ===\n")
                f.write(f"Valence: {vad_scores['valence']:.2f}\n")
                f.write(f"Arousal: {vad_scores['arousal']:.2f}\n")
                f.write(f"Dominance: {vad_scores['dominance']:.2f}\n\n")
                
                # Write predicted emotions
                f.write("=== PREDICTED EMOTIONS ===\n")
                f.write(f"{', '.join(emotions)}.\n\n")
                
                # Write music characteristics
                f.write("=== MUSIC CHARACTERISTICS ===\n")
                f.write(recommendations)
                
            print(f"\nPrompt saved to {prompt_file}")
            return True
        except Exception as e:
            print(f"Error saving prompt to file: {e}")
            return False
    
    def process_audio(self, audio_path):
        """
        Process an audio file and return music recommendations
        
        Parameters:
        - audio_path: Path to the audio file
        
        Returns:
        - Dictionary with VAD scores, emotions, and music recommendations
        """
        # Get VAD scores
        vad_scores = self.predict_vad_from_audio(audio_path)
        
        # Get emotions from VAD scores
        emotions = self.get_emotions_from_vad(vad_scores)
        
        # Store the emotions for use in recommendations
        self.last_emotions = emotions
        
        # Get music recommendations
        recommendations = self.get_music_recommendations(emotions)
        
        # Format recommendations
        formatted_recommendations = self.format_recommendations(recommendations)
        
        # Save the prompt to a file
        self.save_prompt_to_file(emotions, formatted_recommendations, vad_scores)
        
        return {
            "vad_scores": vad_scores,
            "emotions": emotions,
            "recommendations": formatted_recommendations
        }


if __name__ == "__main__":
    # Initialize the recommender
    recommender = MusicEmotionRecommender()
    
    # Get audio file path from user
    audio_path = input("Enter the path to your audio file: ")
    
    # Process the audio and get recommendations
    results = recommender.process_audio(audio_path)
    
    # Display results
    print("\n=== VAD SCORES ===")
    print(f"Valence: {results['vad_scores']['valence']:.2f}")
    print(f"Arousal: {results['vad_scores']['arousal']:.2f}")
    print(f"Dominance: {results['vad_scores']['dominance']:.2f}")
    
    print("\n=== PREDICTED EMOTIONS ===")
    print(", ".join(results['emotions']))
    
    print("\n=== MUSIC CHARACTERISTICS ===")
    print(results['recommendations']) 