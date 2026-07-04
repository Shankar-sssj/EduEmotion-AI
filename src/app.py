import os
import sys
import time
import pandas as pd
import plotly.express as px
import streamlit as st

# Add the src folder to path to import helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier import BiLSTMClassifier, BERTClassifier, RuleBasedClassifier, get_fallback_model
from gemini_helper import generate_personalized_guidance
from utils import DATA_DIR, LOGS_FILE_PATH, MODELS_DIR, STUDENT_DATASET_PATH, log_interaction, get_logs_df, download_nltk_dependencies, initialize_logs

# Set page config
st.set_page_config(
    page_title="EduEmotion AI - Personalized Learning Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    /* Core variables and resets */
    :root {
        --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        --card-bg: rgba(30, 41, 59, 0.7);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --text-main: #f8fafc;
        --text-sub: #94a3b8;
    }
    
    /* Main body background override */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 50%, #030712 100%) !important;
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Headers styling */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
    }
    
    /* Premium card container styling */
    .emotion-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .emotion-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 10px 35px rgba(99, 102, 241, 0.1);
    }
    
    /* Emotion Badge configurations */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        margin-bottom: 12px;
        text-transform: uppercase;
        border: 1px solid transparent;
    }
    
    .badge-bored { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border-color: rgba(100, 116, 139, 0.4); }
    .badge-confident { background: rgba(16, 185, 129, 0.2); color: #34d399; border-color: rgba(16, 185, 129, 0.4); }
    .badge-confused { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border-color: rgba(245, 158, 11, 0.4); }
    .badge-curious { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border-color: rgba(59, 130, 246, 0.4); }
    .badge-frustrated { background: rgba(239, 68, 68, 0.2); color: #f87171; border-color: rgba(239, 68, 68, 0.4); }
    
    /* Subtext styling */
    .subtext {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Sidebar premium overlay */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Custom divider */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent);
        margin: 20px 0;
        opacity: 0.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logs CSV file on load
initialize_logs()

# Cache Classifiers for efficiency
@st.cache_resource
def load_classifiers():
    bilstm = BiLSTMClassifier()
    bert = BERTClassifier()
    rules = RuleBasedClassifier()
    return bilstm, bert, rules

bilstm_clf, bert_clf, rules_clf = load_classifiers()

# Sidebar Setup
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0px;'>EduEmotion AI</h1>
        <p style='color: #818cf8; font-weight: 500; font-size: 0.95rem; margin-top: 5px;'>Emotion-Aware Study Support</p>
    </div>
    <div class='glow-divider'></div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.subheader("🔑 Gemini AI API Configuration")
api_key = st.sidebar.text_input(
    "API Key", 
    type="password", 
    placeholder="Enter Gemini API key here...",
    help="Provide your Gemini API key to receive actual real-time guidance. If empty, the app runs in demo mode with pre-configured guidance."
)

if api_key:
    st.sidebar.success("Gemini API Key active!")
else:
    st.sidebar.info("Running in **Demo Mode** (Simulated AI guidance enabled)")

st.sidebar.markdown("---")

st.sidebar.subheader("🔬 Model Settings")
classifier_choice = st.sidebar.selectbox(
    "Primary Assistant Model",
    ["Scikit-Learn Fallback Model", "BiLSTM Neural Network", "BERT Transformer (Zero-Shot)"],
    index=0,
    help="Select which classifier determines the primary emotion for AI generation. BiLSTM requires training first."
)

# Sidebar System Health Status Check
st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Environment Status")

tf_loaded = False
try:
    import tensorflow
    tf_loaded = True
except ImportError:
    pass

torch_loaded = False
transformers_loaded = False
try:
    import torch
    import transformers
    torch_loaded = True
    transformers_loaded = True
except ImportError:
    pass

bilstm_trained = os.path.exists(os.path.join(MODELS_DIR, "bilstm_model.keras"))

status_items = [
    ("TensorFlow (Python 3.13+)", "🟢 Installed" if tf_loaded else "🔴 Missing (Using Fallback)"),
    ("PyTorch / HF", "🟢 Installed" if torch_loaded else "🔴 Missing (Using Fallback)"),
    ("BiLSTM Trained Weights", "🟢 Available" if bilstm_trained else "🔴 Not Trained"),
]

for name, status in status_items:
    st.sidebar.markdown(f"**{name}**: {status}")

st.sidebar.markdown(
    """
    <div class='glow-divider'></div>
    <div style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 15px;'>
        EduEmotion AI v1.0.0
        <br>Built for student emotional wellness
    </div>
    """, 
    unsafe_allow_html=True
)

# App Header
col_header_title, col_header_stats = st.columns([3, 1])
with col_header_title:
    st.title("🎓 Emotion-Aware Personalized Learning Support")
    st.write("An end-to-end sentiment framework that translates student frustrations and study challenges into personalized, empathetic learning roadmaps.")
with col_header_stats:
    logs_df = get_logs_df()
    total_logs = len(logs_df) if not logs_df.empty else 0
    st.metric("Total Logged Interactions", total_logs)

# Create Page Navigation Tabs
tab_assistant, tab_comparison, tab_analytics, tab_logs, tab_settings = st.tabs([
    "💬 Student Assistant",
    "⚖️ Side-by-Side Model Comparison",
    "📈 Analytics Dashboard",
    "📂 Saved CSV Logs",
    "⚙️ Training & System Config"
])

# Icon and badge mapping helper
emotion_meta = {
    "Bored": {"icon": "🥱", "badge": "badge-bored", "color": "#64748b", "tagline": "Unmotivated & disengaged"},
    "Confident": {"icon": "🚀", "badge": "badge-confident", "color": "#10b981", "tagline": "Mastering the topic"},
    "Confused": {"icon": "🔍", "badge": "badge-confused", "color": "#f59e0b", "tagline": "Needs clarity & visual aids"},
    "Curious": {"icon": "💡", "badge": "badge-curious", "color": "#3b82f6", "tagline": "Seeking deeper insight"},
    "Frustrated": {"icon": "🛠️", "badge": "badge-frustrated", "color": "#ef4444", "tagline": "Stuck & hitting obstacles"}
}

# ==================== TAB 1: STUDENT ASSISTANT ====================
with tab_assistant:
    st.header("💬 Real-Time Learning Assistant")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("What subject or study block are you struggling with?")
        student_text = st.text_area(
            "Describe your coding bug, math problem, or homework frustration in plain words:",
            value="I've been stuck on this recursion code for hours and my base case keeps causing a stack overflow! I'm so close to giving up.",
            height=130,
            placeholder="Type your study frustration or exploration query here...",
            key="input_text"
        )
        
        # Trigger prediction on button click
        submit_btn = st.button("🚀 Analyze Emotion & Generate Guide", type="primary")
        
    with col2:
        st.markdown(
            """
            <div class='emotion-card'>
                <h4>💡 Tip for Writing Inputs</h4>
                <p class='subtext'>The assistant detects learning states like <b>Frustrated</b>, <b>Confused</b>, <b>Bored</b>, <b>Curious</b>, or <b>Confident</b>.</p>
                <p class='subtext'>Try pasting:</p>
                <ul style='color: #94a3b8; font-size: 0.85rem; padding-left: 1.2rem;'>
                    <li><i>"I don't get the difference between pointers and references, what is going on?"</i> (Confused)</li>
                    <li><i>"This math theorem is so dry, it feels completely useless."</i> (Bored)</li>
                    <li><i>"How does memory allocation work under the hood?"</i> (Curious)</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Process input
    if submit_btn and student_text:
        with st.spinner("Analyzing emotion pipeline & calling Gemini AI..."):
            # 1. Classify emotion using primary model choice
            if classifier_choice == "BiLSTM Neural Network":
                pred_result = bilstm_clf.predict(student_text)
            elif classifier_choice == "BERT Transformer (Zero-Shot)":
                pred_result = bert_clf.predict(student_text)
            else:
                # Fallback model: use get_fallback_model
                fallback = get_fallback_model()
                if fallback:
                    vec, clf = fallback
                    feat = vec.transform([student_text])
                    probs = clf.predict_proba(feat)[0]
                    classes = clf.classes_
                    scores = {c: float(p) for c, p in zip(classes, probs)}
                    label = max(scores, key=scores.get)
                    pred_result = {"label": label, "scores": scores, "method": "Fallback TF-IDF"}
                else:
                    pred_result = rules_clf.predict(student_text)
            
            primary_emotion = pred_result["label"]
            scores = pred_result["scores"]
            method_used = pred_result["method"]
            primary_confidence = scores.get(primary_emotion, 0.0)
            
            # 2. Get gemini guidance response
            ai_guidance = generate_personalized_guidance(
                student_text, primary_emotion, primary_confidence, api_key=api_key
            )
            
            # Log interaction to CSV file
            log_interaction(student_text, primary_emotion, primary_confidence, method_used, ai_guidance)
            
            # Display results in columns
            col_res_left, col_res_right = st.columns([1, 1])
            
            with col_res_left:
                st.markdown("### 🎯 Classification Result")
                meta = emotion_meta.get(primary_emotion, {"icon": "❓", "badge": "badge-bored", "color": "#ccc", "tagline": ""})
                
                # HTML Card showing predicted emotion
                st.markdown(
                    f"""
                    <div class='emotion-card' style='border-left: 5px solid {meta["color"]}'>
                        <span class='badge {meta["badge"]}'>{primary_emotion}</span>
                        <h2>{meta["icon"]} {primary_emotion}</h2>
                        <p style='color: #f8fafc; font-style: italic;'>"{meta["tagline"]}"</p>
                        <p class='subtext'>Classifier: <b>{method_used}</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Plotly breakdown for this specific model prediction
                prob_df = pd.DataFrame({
                    "Emotion": list(scores.keys()),
                    "Probability": list(scores.values())
                }).sort_values(by="Probability", ascending=True)
                
                fig = px.bar(
                    prob_df, 
                    x="Probability", 
                    y="Emotion", 
                    orientation='h',
                    title="Emotion Breakdown (Confidence Score)",
                    color="Emotion",
                    color_discrete_map={k: v["color"] for k, v in emotion_meta.items()}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#94a3b8',
                    height=260,
                    showlegend=False,
                    xaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True, key="assistant_emotion_breakdown")
                
            with col_res_right:
                st.markdown("### 🤖 Personalized AI Guidance")
                st.markdown(
                    f"""
                    <div class='emotion-card' style='background-color: rgba(99, 102, 241, 0.08); border-color: rgba(99, 102, 241, 0.25); min-height: 400px;'>
                        {ai_guidance}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.toast(f"Logged emotion prediction: {primary_emotion} ({primary_confidence:.1%})", icon="💾")

# ==================== TAB 2: SIDE-BY-SIDE MODEL COMPARISON ====================
with tab_comparison:
    st.header("⚖️ Dual-Model & Pipeline Comparison")
    st.write("Review and compare the sentiment classification differences, confidence levels, and latency speeds across models.")
    
    comp_text = st.text_input(
        "Enter test sentence for model verification:",
        value="I want to learn more about the advanced memory management inside pointers and references, is there a project I can build?",
        key="comp_text"
    )
    
    if st.button("⚖️ Run Comparative Inference"):
        with st.spinner("Processing side-by-side analysis..."):
            
            # Run inference on all models
            # 1. Fallback Classifier
            t0 = time.time()
            fallback = get_fallback_model()
            if fallback:
                vec, clf = fallback
                feat = vec.transform([comp_text])
                probs = clf.predict_proba(feat)[0]
                classes = clf.classes_
                fb_scores = {c: float(p) for c, p in zip(classes, probs)}
                fb_pred = max(fb_scores, key=fb_scores.get)
            else:
                fb_scores = {k: 0.2 for k in emotion_meta.keys()}
                fb_pred = "Confused"
            t_fb = (time.time() - t0) * 1000 # ms
            
            # 2. BiLSTM Model
            t0 = time.time()
            bilstm_res = bilstm_clf.predict(comp_text)
            t_bilstm = (time.time() - t0) * 1000 # ms
            
            # 3. BERT Zero-shot Model
            t0 = time.time()
            bert_res = bert_clf.predict(comp_text)
            t_bert = (time.time() - t0) * 1000 # ms
            
            # 4. Rules Keyword Model
            t0 = time.time()
            rules_res = rules_clf.predict(comp_text)
            t_rules = (time.time() - t0) * 1000 # ms
            
            col_col1, col_col2, col_col3, col_col4 = st.columns(4)
            
            models_info = [
                ("Fallback ML", fb_pred, fb_scores, t_fb, col_col1, "TF-IDF + Logistic Reg"),
                ("BiLSTM", bilstm_res["label"], bilstm_res["scores"], t_bilstm, col_col2, "Deep Learning (TensorFlow)"),
                ("BERT Transformer", bert_res["label"], bert_res["scores"], t_bert, col_col3, "Transformers Zero-Shot"),
                ("Keyword Rule-Based", rules_res["label"], rules_res["scores"], t_rules, col_col4, "Custom Phrase Matcher")
            ]
            
            for name, pred, scores, latency, column, details in models_info:
                with column:
                    meta = emotion_meta.get(pred, {"icon": "❓", "badge": "badge-bored", "color": "#ccc"})
                    st.markdown(
                        f"""
                        <div class='emotion-card' style='border-top: 4px solid {meta["color"]}; min-height: 180px;'>
                            <h4 style='margin-bottom:2px;'>{name}</h4>
                            <p style='color:#64748b; font-size:0.8rem; margin-top:0px;'>{details}</p>
                            <span class='badge {meta["badge"]}'>{pred}</span>
                            <div style='margin-top:10px;'>
                                <span style='font-size:0.8rem; color:#94a3b8;'>Inference: <b>{latency:.2f} ms</b></span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Plot horizontal chart for individual confidence breakdown
                    prob_df_c = pd.DataFrame({
                        "Emotion": list(scores.keys()),
                        "Probability": list(scores.values())
                    }).sort_values(by="Probability", ascending=True)
                    
                    fig_c = px.bar(
                        prob_df_c, 
                        x="Probability", 
                        y="Emotion", 
                        orientation='h',
                        height=200,
                        color="Emotion",
                        color_discrete_map={k: v["color"] for k, v in emotion_meta.items()}
                    )
                    fig_c.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8',
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False,
                        xaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_c, use_container_width=True, key=f"comparison_{name.replace(' ', '_')}")

# ==================== TAB 3: ANALYTICS DASHBOARD ==================
with tab_analytics:
    st.header("📈 Historical Emotion Trends & Logs Analytics")
    
    logs_df = get_logs_df()
    
    if logs_df.empty:
        st.info("No interactive logs found yet. Enter problems in the 'Student Assistant' tab to generate tracking analytics!")
    else:
        # Layout metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Logs", len(logs_df))
        with col_m2:
            st.metric("Average Confidence", f"{logs_df['Confidence_Score'].mean():.1%}")
        with col_m3:
            dominant_overall = logs_df['Predicted_Emotion'].value_counts().index[0]
            st.metric("Top Emotion", f"{dominant_overall} ({emotion_meta[dominant_overall]['icon']})")
            
        st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
        
        # Plot distribution charts
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            # Bar chart of emotions
            counts_df = logs_df['Predicted_Emotion'].value_counts().reset_index()
            counts_df.columns = ["Emotion", "Count"]
            fig_bar = px.bar(
                counts_df,
                x="Count",
                y="Emotion",
                orientation='h',
                title="Total Emotion Distribution",
                color="Emotion",
                color_discrete_map={k: v["color"] for k, v in emotion_meta.items()}
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_bar, use_container_width=True, key="analytics_emotion_distribution")
            
        with col_plot2:
            # Pie chart of models
            model_counts = logs_df['Model_Used'].value_counts().reset_index()
            model_counts.columns = ["Model", "Logs Count"]
            fig_pie = px.pie(
                model_counts,
                values="Logs Count",
                names="Model",
                title="Prediction Models Utilized",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_pie, use_container_width=True, key="analytics_model_usage_pie")
            
        # Temporal analysis
        logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
        logs_df['Date'] = logs_df['Timestamp'].dt.date
        date_counts = logs_df.groupby(['Date', 'Predicted_Emotion']).size().reset_index(name='Count')
        
        fig_line = px.line(
            date_counts,
            x="Date",
            y="Count",
            color="Predicted_Emotion",
            title="Temporal Student Emotion Trends (Daily Activity)",
            color_discrete_map={k: v["color"] for k, v in emotion_meta.items()}
        )
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
        st.plotly_chart(fig_line, use_container_width=True, key="analytics_trend_line")

# ==================== TAB 4: SAVED LOGS VIEW ==================
with tab_logs:
    st.header("📂 Student Interaction Logs CSV")
    
    logs_df = get_logs_df()
    
    if logs_df.empty:
        st.info("Log file is currently empty.")
    else:
        st.subheader("Filter and Search Interactions")
        
        search_query = st.text_input("Search student query content:", "")
        emotion_filter = st.multiselect("Filter by Emotion:", list(emotion_meta.keys()))
        
        # Apply filters
        filtered_df = logs_df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df['Student_Text'].str.contains(search_query, case=False, na=False)]
        if emotion_filter:
            filtered_df = filtered_df[filtered_df['Predicted_Emotion'].isin(emotion_filter)]
            
        st.dataframe(
            filtered_df.sort_values(by="Timestamp", ascending=False),
            use_container_width=True
        )
        
        # Clear Logs Button
        if st.button("🗑️ Clear All Logs"):
            if os.path.exists(LOGS_FILE_PATH):
                os.remove(LOGS_FILE_PATH)
                initialize_logs()
                st.success("CSV logs cleared successfully! Reload the page.")
                st.rerun()

# ==================== TAB 5: SYSTEM CONFIG & RUNNERS ====================
with tab_settings:
    st.header("⚙️ Training and Pipeline Configuration")
    
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        st.subheader("🛠️ Model Training Panel")
        st.write("Generate training datasets and retrain the Keras BiLSTM neural network directly from this interface.")
        
        # Run dataset generator button
        if st.button("1. Generate Training Dataset (student_dataset.csv)"):
            from dataset_generator import generate_synthetic_dataset
            df = generate_synthetic_dataset()
            st.success(f"Generated synthetic training data with {len(df)} samples!")
            
        # Run BiLSTM training button
        train_btn = st.button("2. Train Keras BiLSTM Model")
        if train_btn:
            with st.spinner("Training BiLSTM model... Check terminal execution"):
                from train_bilstm import train_bilstm_model
                success = train_bilstm_model()
                if success:
                    st.success("Keras BiLSTM model trained and saved to `models/bilstm_model.keras`!")
                    st.cache_resource.clear() # Clear cache to load new model
                    st.rerun()
                else:
                    st.error("Training failed. Check console outputs and ensure TensorFlow is installed.")
                    
    with col_cfg2:
        st.subheader("📚 Dataset Information")
        st.write("Review the size and samples of the generated training dataset.")
        
        csv_path = STUDENT_DATASET_PATH
        if os.path.exists(csv_path):
            df_train = pd.read_csv(csv_path)
            st.write(f"**Dataset status**: Generated and available.")
            st.write(f"**Total sample count**: {len(df_train)} rows")
            
            # Show class distribution
            train_dist = df_train['emotion'].value_counts().reset_index()
            train_dist.columns = ["Emotion", "Train Samples"]
            fig_dist = px.bar(
                train_dist, 
                x="Emotion", 
                y="Train Samples", 
                color="Emotion",
                color_discrete_map={k: v["color"] for k, v in emotion_meta.items()}
            )
            fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=240, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True, key="settings_train_dist")
        else:
            st.warning("Training dataset not found. Click the button to generate the dataset!")
            
    st.markdown("<div class='glow-divider'></div>", unsafe_allow_html=True)
    st.subheader("📝 Kaggle & Cloud GPU Integration Instruction")
    st.markdown(
        """
        To train a heavy BERT model (e.g., fine-tuning `bert-base-uncased` on educational datasets) or run heavy epochs of BiLSTM with CUDA acceleration:
        1. Access the notebook script created in the `notebooks/Kaggle_Emotion_Training.ipynb` directory.
        2. Upload the `data/student_dataset.csv` to Kaggle as a dataset.
        3. Copy the notebook cells into a Kaggle Notebook, activate GPU T4 x2 or P100.
        4. Run training, and download the resulting `bilstm_model.keras` or BERT model outputs back into your local project files!
        """
    )
