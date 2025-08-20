#!/usr/bin/env python3
"""Train a transformer-based intent classification model from intents_dataset.json.

Produces: AI_bot/intent_transformer.h5 and AI_bot/intent_transformer_resources.joblib

Usage: python train_transformer.py --epochs 50 --batch-size 16
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


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


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)

    # Feed forward network
    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    ffn_output = Add()([attention_output, ffn_output])
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output)
    
    return ffn_output


def build_transformer_model(
    vocab_size, 
    max_len, 
    embed_dim=128, 
    num_heads=4, 
    ff_dim=128, 
    num_blocks=2, 
    dropout=0.1,
    num_classes=7
):
    inputs = Input(shape=(max_len,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, embed_dim)(inputs)
    embedding = Dropout(dropout)(embedding)
    
    # Multiple transformer blocks
    x = embedding
    for _ in range(num_blocks):
        x = transformer_encoder_block(
            x, head_size=embed_dim // num_heads, num_heads=num_heads, 
            ff_dim=ff_dim, dropout=dropout
        )
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    
    # Classification head
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model


def build_and_train(
    texts,
    labels,
    max_words=8000,
    max_len=50,
    embed_dim=128,
    epochs=50,
    batch_size=16,
    num_heads=4,
    ff_dim=128,
    num_blocks=2,
    dropout=0.1,
    patience=15,
    val_split=0.15,
    learning_rate=0.001,
    use_class_weight=True,
    use_early_stopping=True,
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

    model = build_transformer_model(
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        num_classes=num_classes
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )

    print("Kiến trúc mô hình Transformer:")
    model.summary()

    # Callbacks
    rlrop = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=max(3, patience // 3), verbose=1
    )
    
    callbacks = [rlrop]
    if use_early_stopping:
        es = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        callbacks.insert(0, es)

    # Class weights
    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
        print("Sử dụng class_weight:", class_weight)

    # Training
    history = model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Evaluation
    preds_probs = model.predict(X_test)
    preds = np.argmax(preds_probs, axis=1)
    acc = accuracy_score(y_test, preds)
    print(f"Độ chính xác trên holdout: {acc:.4f}")
    print("Báo cáo phân loại:\n", classification_report(y_test, preds, target_names=le.classes_))

    return model, tokenizer, le, history


def save_model(model, tokenizer, label_encoder, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "intent_transformer.h5"
    resources_path = out_dir / "intent_transformer_resources.joblib"
    
    model.save(model_path)
    joblib.dump({"tokenizer": tokenizer, "label_encoder": label_encoder}, resources_path)
    
    print(f"Đã lưu mô hình Transformer tới {model_path}")
    print(f"Đã lưu tokenizer và label encoder tới {resources_path}")


def main():
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--max-len", type=int, default=50, help="Độ dài sequence tối đa")
    parser.add_argument("--max-words", type=int, default=8000, help="Số từ tối đa cho tokenizer")
    parser.add_argument("--embed-dim", type=int, default=128, help="Chiều embedding")
    parser.add_argument("--num-heads", type=int, default=4, help="Số attention heads")
    parser.add_argument("--ff-dim", type=int, default=128, help="Feed forward dimension")
    parser.add_argument("--num-blocks", type=int, default=2, help="Số transformer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Tỷ lệ dropout")
    parser.add_argument("--patience", type=int, default=15, help="Patience cho EarlyStopping")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate ban đầu")
    parser.add_argument("--val-split", type=float, default=0.15, help="Tỷ lệ validation split")
    parser.add_argument("--no-class-weight", dest="use_class_weight", action="store_false", help="Tắt class weighting")
    parser.add_argument("--no-early-stopping", dest="use_early_stopping", action="store_false", help="Tắt EarlyStopping")
    
    args = parser.parse_args()

    dataset_path = base / "intents_dataset.json"
    if not dataset_path.exists():
        raise SystemExit(f"Không tìm thấy dataset: {dataset_path}")

    texts, labels = load_dataset(dataset_path)
    if not texts:
        raise SystemExit("Không tìm thấy dữ liệu huấn luyện trong dataset")

    model, tokenizer, le, history = build_and_train(
        texts,
        labels,
        max_words=args.max_words,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        patience=args.patience,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        use_class_weight=args.use_class_weight,
        use_early_stopping=args.use_early_stopping,
    )
    
    save_model(model, tokenizer, le, base)


if __name__ == "__main__":
    main()
