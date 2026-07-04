import os
import csv
import datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_FILE_PATH = os.path.join(DATA_DIR, "interaction_logs.csv")
STUDENT_DATASET_PATH = os.path.join(DATA_DIR, "student_dataset.csv")


def initialize_logs():
    """Creates the data directory and the logs CSV file if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(LOGS_FILE_PATH):
        with open(LOGS_FILE_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", 
                "Student_Text", 
                "Predicted_Emotion", 
                "Confidence_Score", 
                "Model_Used", 
                "Gemini_Response"
            ])
        print(f"Initialized logs CSV at {LOGS_FILE_PATH}")

def log_interaction(text, emotion, confidence, model_name, response):
    """Logs an interaction to the CSV file."""
    initialize_logs()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(LOGS_FILE_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                text,
                emotion,
                round(confidence, 4),
                model_name,
                response
            ])
        return True
    except Exception as e:
        print(f"Failed to log interaction to CSV: {e}")
        return False

def get_logs_df():
    """Reads logs CSV and returns a pandas DataFrame."""
    initialize_logs()
    try:
        df = pd.read_csv(LOGS_FILE_PATH)
        return df
    except Exception as e:
        print(f"Error reading logs: {e}")
        return pd.DataFrame()

def download_nltk_dependencies():
    """Downloads necessary NLTK components silently."""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        return True
    except ImportError:
        return False
