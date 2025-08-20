#!/usr/bin/env python3
"""Train an intent classification LSTM model from `intents_dataset.json`.

Produces: AI_bot/intent_lstm.h5 and AI_bot/intent_resources.joblib

Usage: python AI_bot/train_intent.py --epochs 10 --batch-size 32
"""
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    labels = []
    for entry in data:
        t = entry.get("text")
        intent = entry.get("intent")
        if not t or not intent:
            continue
        texts.append(t)
        labels.append(intent)
    return texts, labels


def build_and_train(
    texts,
    labels,
    max_words=8000,
    max_len=50,
    embed_dim=200,
    epochs=150,
    batch_size=16,
    lstm_units=128,
    dropout=0.3,
    recurrent_dropout=0.2,
    patience=30,
    val_split=0.15,
    learning_rate=0.0005,
    use_class_weight=True,
    use_early_stopping=True,  # <-- thêm tham số này
):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # print class distribution for debugging
    unique, counts = np.unique(y_train, return_counts=True)
    print("Phân phối lớp trong tập train:")
    for u, c in zip(unique, counts):
        print(f"  lớp {u}: {c}")

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout),
        Dropout(dropout),
        Dense(max(64, lstm_units // 2), activation="relu"),
        Dropout(dropout / 2),
        Dense(num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, patience // 3), verbose=1)

    # build callbacks list conditionally
    callbacks = [rlrop]
    if use_early_stopping:
        es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        callbacks.insert(0, es)

    # compute class weights to mitigate imbalance (optional)
    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print("Sử dụng class_weight:", class_weight)

    model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,  # <-- dùng callbacks thay vì [es, rlrop]
        class_weight=class_weight,
        verbose=1,
    )

    preds_probs = model.predict(X_test)
    preds = np.argmax(preds_probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Độ chính xác trên holdout: {acc:.4f}")
    print("Báo cáo phân loại:\n", classification_report(y_test, preds, target_names=le.classes_))

    return model, tokenizer, le


def save_model(model, tokenizer, label_encoder, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "intent_lstm.h5"
    resources_path = out_dir / "intent_resources.joblib"
    model.save(model_path)
    joblib.dump({"tokenizer": tokenizer, "label_encoder": label_encoder}, resources_path)
    print(f"Đã lưu mô hình Keras tới {model_path}")
    print(f"Đã lưu tokenizer và label encoder tới {resources_path}")


def main():
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs (default tuned for ~800 samples)")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max-len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--max-words", type=int, default=8000, help="Max words for tokenizer")
    parser.add_argument("--embed-dim", type=int, default=200, help="Embedding dimension")
    parser.add_argument("--lstm-units", type=int, default=128, help="Number of LSTM units")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--recurrent-dropout", type=float, default=0.2, help="Recurrent dropout rate")
    parser.add_argument("--patience", type=int, default=30, help="EarlyStopping patience")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split from training data")
    parser.add_argument("--no-class-weight", dest="use_class_weight", action="store_false", help="Disable class weighting")
    args = parser.parse_args()

    dataset_path = base / "intents_dataset.json"
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    texts, labels = load_dataset(dataset_path)
    if not texts:
        raise SystemExit("No training data found in dataset")

    model, tokenizer, le = build_and_train(
        texts,
        labels,
        max_words=args.max_words,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        patience=args.patience,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        use_class_weight=args.use_class_weight,
    )
    save_model(model, tokenizer, le, base)


if __name__ == "__main__":
    main()
