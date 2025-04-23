from transformers import pipeline
import torch
import re
import math

# Instead of loading a specific model, use a more comprehensive rule-based approach
# This handles a diverse range of emotions with better precision

# Mapping for common misformats
emotion_variants = {
    "Anger": "Angry",
    "Excitement": "Excited",
    "Happiness": "Happy",
    "Sadness": "Sad",
    "Fear": "Fearful",
    "Disgust": "Disgusted",
    "Surprise": "Surprised",
    "Tiredness": "Tired",
    "Boredom": "Bored",
    "Confusion": "Confused",
    "Neutrality": "Neutral",
    "Content": "Content",
    "Calm": "Calm",
    "Relaxed": "Relaxed",
    "Anxious": "Anxious",
    "Frustration": "Frustrated",
    "Awe": "Awestruck",
    "Nostalgia": "Nostalgic",
    "Melancholy": "Melancholic",
    "Passion": "Passionate"
}

# Full emotion set
valid_emotions = set([
    "Happy", "Sad", "Angry", "Fearful", "Disgusted",
    "Surprised", "Neutral", "Tired", "Bored", "Confused", 
    "Excited", "Content", "Calm", "Relaxed", "Anxious",
    "Frustrated", "Awestruck", "Nostalgic", "Melancholic", "Passionate"
])

# Emotions with their VAD ranges (min_v, max_v, min_a, max_a, min_d, max_d)
# Based on research in affective computing 
emotion_vad_ranges = {
    "Happy": (3.5, 5.0, 3.0, 5.0, 3.0, 5.0),
    "Excited": (3.5, 5.0, 4.0, 5.0, 3.5, 5.0),
    "Content": (3.5, 5.0, 1.0, 3.0, 3.0, 5.0),
    "Relaxed": (3.0, 4.5, 1.0, 2.5, 3.0, 4.5),
    "Calm": (3.0, 4.0, 1.0, 2.5, 3.0, 4.5),
    "Neutral": (2.5, 3.5, 2.0, 3.0, 2.5, 3.5),
    "Bored": (2.0, 3.0, 1.0, 2.0, 2.0, 3.0),
    "Tired": (2.0, 3.0, 1.0, 2.5, 1.0, 3.0),
    "Sad": (1.0, 2.5, 1.0, 3.0, 1.0, 3.0),
    "Melancholic": (2.0, 3.0, 1.5, 3.0, 2.0, 3.5),
    "Nostalgic": (3.0, 4.0, 2.0, 3.0, 2.5, 4.0),
    "Anxious": (1.5, 3.0, 3.5, 5.0, 1.0, 2.5),
    "Fearful": (1.0, 2.5, 3.5, 5.0, 1.0, 2.5),
    "Confused": (2.0, 3.0, 2.5, 4.0, 1.0, 2.5),
    "Frustrated": (1.5, 2.5, 3.5, 5.0, 2.5, 4.0),
    "Angry": (1.0, 2.5, 3.5, 5.0, 3.5, 5.0),
    "Disgusted": (1.0, 2.0, 3.0, 4.5, 2.5, 4.0),
    "Surprised": (2.5, 4.0, 3.5, 5.0, 2.0, 4.0),
    "Awestruck": (3.5, 5.0, 3.0, 4.5, 2.0, 3.5),
    "Passionate": (3.5, 5.0, 3.5, 5.0, 4.0, 5.0)
}

def normalize_emotion(word):
    word = word.strip().capitalize()
    return emotion_variants.get(word, word) if word in emotion_variants else word

def emotion_vad_match_score(valence, arousal, dominance, emotion):
    """
    Calculate how well a VAD score matches an emotion based on its expected VAD range
    Returns a similarity score (0-1) where 1 is a perfect match
    """
    if emotion not in emotion_vad_ranges:
        return 0.0
        
    min_v, max_v, min_a, max_a, min_d, max_d = emotion_vad_ranges[emotion]
    
    # Calculate match for each dimension
    v_match = 1.0 - min(1.0, abs(valence - (min_v + max_v) / 2) / ((max_v - min_v) / 2 + 0.5))
    a_match = 1.0 - min(1.0, abs(arousal - (min_a + max_a) / 2) / ((max_a - min_a) / 2 + 0.5))
    d_match = 1.0 - min(1.0, abs(dominance - (min_d + max_d) / 2) / ((max_d - min_d) / 2 + 0.5))
    
    # Check if within range (with a small buffer)
    in_range_v = min_v - 0.3 <= valence <= max_v + 0.3
    in_range_a = min_a - 0.3 <= arousal <= max_a + 0.3
    in_range_d = min_d - 0.3 <= dominance <= max_d + 0.3
    
    # Calculate overall match score
    base_match = (v_match + a_match + d_match) / 3
    
    # Apply a penalty if any dimension is completely out of range
    if not (in_range_v and in_range_a and in_range_d):
        base_match *= 0.7
    
    return base_match

def predict_emotions_from_vad(valence, arousal, dominance):
    """
    Predict emotions based on VAD scores using an enhanced rule-based approach
    
    Parameters:
    - valence: Valence score (1-5 scale)
    - arousal: Arousal score (1-5 scale)
    - dominance: Dominance score (1-5 scale)
    
    Returns:
    - String with the top 3 emotions
    """
    # Calculate match scores for all emotions
    emotion_scores = {}
    for emotion in emotion_vad_ranges.keys():
        score = emotion_vad_match_score(valence, arousal, dominance, emotion)
        emotion_scores[emotion] = score
    
    # Get the top emotions by score
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Debug information (for development)
    # for emotion, score in sorted_emotions[:6]:
    #     print(f"{emotion}: {score:.2f}")
    
    # Select top emotions
    top_emotions = [emotion for emotion, score in sorted_emotions[:5] if score > 0.5]
    
    # If we don't have enough emotions with good scores, add the next best ones
    if len(top_emotions) < 3:
        additional_emotions = [emotion for emotion, score in sorted_emotions 
                               if emotion not in top_emotions]
        top_emotions.extend(additional_emotions)
    
    # Select exactly 3 emotions
    emotions = top_emotions[:3]
    
    # Ensure we have 3 emotions - add fallbacks if needed
    if len(emotions) < 3:
        fallbacks = ["Neutral", "Surprised", "Confused"]
        for fallback in fallbacks:
            if fallback not in emotions:
                emotions.append(fallback)
                if len(emotions) == 3:
                    break
    
    return f"Based on your current VAD scores, your top emotions are: {', '.join(emotions)}."

def vad_to_circumplex_angle(valence, arousal):
    """Convert VAD values to an angle on the circumplex model for debugging"""
    # Center the coordinates (assuming 1-5 scale)
    v_centered = valence - 3
    a_centered = arousal - 3
    
    # Calculate angle in radians, then convert to degrees
    angle = math.atan2(a_centered, v_centered) * (180 / math.pi)
    
    return angle

if __name__ == "__main__":
    # Test with a range of VAD scores
    test_cases = [
        (4.2, 4.0, 3.8, "Expected: Happy, Excited"),
        (4.0, 2.0, 3.5, "Expected: Content, Relaxed"),
        (2.0, 1.5, 2.0, "Expected: Sad, Tired"),
        (2.0, 4.0, 4.0, "Expected: Angry"),
        (2.0, 4.0, 2.0, "Expected: Fearful, Anxious"),
        (3.0, 3.0, 3.0, "Expected: Neutral"),
        (2.0, 2.0, 2.0, "Expected: Tired, Sad"),
        (1.5, 4.5, 4.5, "Expected: Angry"),
        (4.5, 4.5, 4.5, "Expected: Excited, Passionate")
    ]
    
    # Run test cases
    for v, a, d, expected in test_cases:
        result = predict_emotions_from_vad(v, a, d)
        print(f"VAD({v}, {a}, {d}) → {result}")
        print(f"{expected}\n")
    
    # For interactive testing
    print("\nInteractive test:")
    valence = float(input("Enter valence (1-5): ") or "3.0")
    arousal = float(input("Enter arousal (1-5): ") or "3.0")
    dominance = float(input("Enter dominance (1-5): ") or "3.0")

    result = predict_emotions_from_vad(valence, arousal, dominance)
    angle = vad_to_circumplex_angle(valence, arousal)
    print(f"Circumplex angle: {angle:.1f}°")
    
    if "top emotions are:" in result:
        print("Your predicted emotions are:" + result.split("top emotions are:")[-1])
    else:
        print(result)

