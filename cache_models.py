# File: cache_models.py
from transformers import pipeline
from faster_whisper import WhisperModel
import torch # Make sure torch is imported

print("--- Starting model caching process ---")

# --- Model Configurations ---
MODEL_NAMES = {
    "transcription": "medium.en",
    "translation": "facebook/nllb-200-distilled-600M",
    "simplification": "tuner007/pegasus_paraphrase",
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "emotion": "superb/wav2vec2-base-superb-er" # New model

}

try:
    # 1. Cache Transcription Model
    print("\nCaching Transcription model (medium.en)...")
    WhisperModel(MODEL_NAMES["transcription"], device="cpu", compute_type="int8")
    print("✅ Transcription model cached.")

    # 2. Cache Translation Model
    print("\nCaching Translation model (NLLB)...")
    pipeline("translation", model=MODEL_NAMES["translation"])
    print("✅ Translation model cached.")

    # 3. Cache Simplification Model
    print("\nCaching Simplification model (Pegasus)...")
    pipeline("text2text-generation", model=MODEL_NAMES["simplification"])
    print("✅ Simplification model cached.")
    
    # 4. Cache Sentiment Model
    print("\nCaching Sentiment model (DistilBERT)...")
    pipeline("sentiment-analysis", model=MODEL_NAMES["sentiment"])
    print("✅ Sentiment model cached.")

    print("\nCaching Emotion Recognition model..."); pipeline("audio-classification", model=MODEL_NAMES["emotion"]); print("✅ Cached.")

    print("\n--- All models have been downloaded and cached successfully! ---")

except Exception as e:
    print(f"\n--- An error occurred: {e} ---")
    print("Please check your internet connection and try again.")