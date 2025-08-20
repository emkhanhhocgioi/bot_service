from pathlib import Path
import joblib
import numpy as np
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Support both LSTM and Transformer models
TRANSFORMER_MODEL_PATH = Path(__file__).parent / "intent_transformer.h5"
TRANSFORMER_RESOURCES_PATH = Path(__file__).parent / "intent_transformer_resources.joblib"
LSTM_MODEL_PATH = Path(__file__).parent / "intent_model.joblib"


def load_transformer_model():
    """Load transformer model and resources."""
    if not TRANSFORMER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Transformer model not found: {TRANSFORMER_MODEL_PATH}")
    if not TRANSFORMER_RESOURCES_PATH.exists():
        raise FileNotFoundError(f"Transformer resources not found: {TRANSFORMER_RESOURCES_PATH}")
    
    model = load_model(TRANSFORMER_MODEL_PATH)
    resources = joblib.load(TRANSFORMER_RESOURCES_PATH)
    tokenizer = resources["tokenizer"]
    label_encoder = resources["label_encoder"]
    
    return model, tokenizer, label_encoder


def load_lstm_model():
    """Load legacy LSTM/sklearn model."""
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"LSTM model not found: {LSTM_MODEL_PATH}")
    
    data = joblib.load(LSTM_MODEL_PATH)
    pipeline = data["pipeline"]
    le = data["label_encoder"]
    return pipeline, le


def predict_intent_transformer(text: str, max_len: int = 50) -> Tuple[str, float]:
    """Predict intent using transformer model."""
    model, tokenizer, label_encoder = load_transformer_model()
    
    # Preprocess
    sequences = tokenizer.texts_to_sequences([text])
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    
    # Predict
    probs = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    intent = label_encoder.inverse_transform([idx])[0]
    
    return intent, confidence


def predict_intent_lstm(text: str) -> Tuple[str, float]:
    """Predict intent using legacy LSTM/sklearn model."""
    pipeline, le = load_lstm_model()
    
    try:
        probs = pipeline.predict_proba([text])[0]
        idx = int(probs.argmax())
        intent = le.inverse_transform([idx])[0]
        confidence = float(probs[idx])
    except Exception:
        # fallback to predict
        pred = pipeline.predict([text])[0]
        intent = le.inverse_transform([pred])[0] if hasattr(le, "inverse_transform") else str(pred)
        confidence = 1.0
    
    return intent, confidence


def predict_intent(text: str, use_transformer: bool = True) -> Tuple[str, float]:
    """Predict intent using either transformer or LSTM model.
    
    Args:
        text: Input text to classify
        use_transformer: If True, use transformer model, otherwise use LSTM
    
    Returns:
        (intent_label, confidence) tuple
    """
    if use_transformer:
        try:
            return predict_intent_transformer(text)
        except FileNotFoundError:
            print("Transformer model không tìm thấy, chuyển sang LSTM model...")
            return predict_intent_lstm(text)
    else:
        return predict_intent_lstm(text)


if __name__ == "__main__":
    # quick interactive demo
    print("Loading transformer model...")
    try:
        while True:
            t = input("Text > ").strip()
            if not t:
                break
            label, conf = predict_intent(t, use_transformer=True)
            print(f"Predicted: {label} (confidence={conf:.3f})")
    except FileNotFoundError as e:
        print(f"Model error: {e}")
        print("Hãy chạy train_transformer.py trước để tạo model.")
