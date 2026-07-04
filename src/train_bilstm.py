import os
import pickle
import numpy as np
import pandas as pd

from utils import STUDENT_DATASET_PATH, MODELS_DIR

# We load tensorflow inside main to allow the script to be run/inspected even if TensorFlow is not installed.
def train_bilstm_model():
    print("Initializing BiLSTM training...")
    
    csv_path = STUDENT_DATASET_PATH
    if not os.path.exists(csv_path):
        print("Dataset not found! Generating dataset first...")
        from dataset_generator import generate_synthetic_dataset
        df = generate_synthetic_dataset()
    else:
        df = pd.read_csv(csv_path)
        
    print(f"Loaded dataset with {len(df)} records.")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
        from sklearn.preprocessing import LabelEncoder
    except ImportError as e:
        print(f"TensorFlow or Scikit-Learn not found. Cannot train BiLSTM. Error: {e}")
        print("Please ensure TensorFlow and Scikit-Learn are installed.")
        return False
        
    # Enforce reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['emotion'])
    num_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)
    print(f"Classes found: {class_names}")
    
    # Tokenize text
    vocab_size = 3000
    max_len = 50
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Build model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model (fast on small dataset)
    print("Training model for 12 epochs...")
    X = np.array(padded_sequences)
    y = np.array(df['label'])
    
    history = model.fit(
        X, y,
        epochs=12,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save model and components
    model_path = os.path.join(MODELS_DIR, "bilstm_model.keras")
    model.save(model_path)
    print(f"Saved Keras model to {model_path}")
    
    # Save tokenizer and label encoder
    meta_path = os.path.join(MODELS_DIR, "bilstm_meta.pkl")
    metadata = {
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "max_len": max_len,
        "class_names": class_names
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata (tokenizer & label encoder) to {meta_path}")
    return True

if __name__ == "__main__":
    train_bilstm_model()
