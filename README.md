# EduEmotion AI: Emotion-Aware Personalized Learning Support Platform

Developed by Shankar-sssj.

An end-to-end sentiment framework that translates student study challenges, bug reports, and homework frustrations into personalized, empathetic learning support roadmaps. It combines machine learning classifiers, neural networks, and generative AI guidance.

## Project Structure

```text
emotion-learning-assistant/
├── README.md
├── requirements.txt
├── data/
│   ├── student_dataset.csv       # Generated training dataset
│   └── interaction_logs.csv      # Logged student queries & responses
├── models/
│   ├── bilstm_model.keras        # Trained local BiLSTM model
│   └── bilstm_meta.pkl           # Tokenizer & label metadata
├── notebooks/
│   └── kaggle_emotion_training.py # Copy-pasteable Kaggle GPU notebook
└── src/
    ├── app.py                    # Main Streamlit web application
    ├── classifier.py             # Classification pipeline & fallbacks
    ├── dataset_generator.py      # Script to generate educational training data
    ├── gemini_helper.py          # Gemini AI API connector & fallbacks
    ├── train_bilstm.py           # TensorFlow BiLSTM model training script
    └── utils.py                  # CSV logger and folder setup helper
```

## How to Set Up and Run

### 1. Set Workspace (Recommended)
Before running, set the project directory as your active workspace:
`C:\Users\shank\OneDrive\Desktop\emotion-learning-assistant`

### 2. Install Dependencies
Install all package requirements in your environment:
```bash
pip install -r requirements.txt
```

### 3. Generate Data and Train Model
Open the Streamlit app settings tab OR run these scripts directly from your console:
```bash
# 1. Generate the training dataset
python src/dataset_generator.py

# 2. Train the BiLSTM Neural Network (optional - fallback models run automatically if skipped)
python src/train_bilstm.py
```

### 4. Run the Streamlit Application
Start the interactive dashboard using Streamlit:
```bash
streamlit run src/app.py
```

## Key Features

1. **💬 Learning Assistant**: A student-facing interface to submit problems, view emotion predictions (Bored, Confident, Confused, Curious, Frustrated), and receive structured, empathetic advice.
2. **⚖️ Model Comparison**: Side-by-side performance comparison of TF-IDF ML, BiLSTM, BERT, and Keyword rules.
3. **📈 Analytics Dashboard**: Plotly charts visualizing trends of student affective states, model choices, and active days.
4. **📁 Logging System**: Automatic background storage of student logs to CSV for progress tracing.
5. **🛡️ Graceful Fallback**: Scikit-Learn models and simulated AI triggers ensure full functionality even without an API key or deep learning packages.
