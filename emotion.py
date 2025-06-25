import streamlit as st
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Emotion & Hate Speech Detection", layout="centered")

# Title and instructions
st.title("😊 Emotion & Hate Speech Detection")
st.markdown("Enter a sentence below. The app will detect all emotion types with their confidence scores and identify if any hate speech is present.")

# Cache analyzers to avoid reloading
@st.cache_resource
def load_analyzers():
    emotion_analyzer = create_analyzer(task='emotion', lang='en')
    hate_speech_analyzer = create_analyzer(task='hate_speech', lang='en')
    return emotion_analyzer, hate_speech_analyzer

emotion_analyzer, hate_speech_analyzer = load_analyzers()

# User input
text_input = st.text_area("📝 Enter text here:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Get predictions
        emo_result = emotion_analyzer.predict(text_input)
        hate_result = hate_speech_analyzer.predict(text_input)

        # Extract highest probability hate label
        hate_label = max(hate_result.probas, key=hate_result.probas.get)
        hate_conf = hate_result.probas[hate_label]

        # Display results
        st.subheader("🧠 Emotion Detection")
        st.write(f"**Top Emotion:** {emo_result.output}")
        st.write("**All Emotion Probabilities:**")
        for label, prob in emo_result.probas.items():
            st.write(f"- {label}: {round(prob * 100, 2)}%")

        # Bar chart for emotions
        st.subheader("📊 Emotion Confidence Scores")
        fig1, ax1 = plt.subplots()
        ax1.bar(emo_result.probas.keys(), emo_result.probas.values(), color='skyblue')
        ax1.set_ylabel("Confidence")
        ax1.set_xlabel("Emotion")
        ax1.set_title("All Emotion Predictions")
        st.pyplot(fig1)

        # Hate speech
        st.subheader("🚫 Hate Speech Detection")
        st.write(f"**Top Hate Speech Category:** {hate_label} ({round(hate_conf * 100, 2)}%)")
        st.write("**All Hate Speech Probabilities:**")
        for label, prob in hate_result.probas.items():
            st.write(f"- {label}: {round(prob * 100, 2)}%")

        # Bar chart for hate speech
        st.subheader("📊 Hate Speech Confidence Scores")
        fig2, ax2 = plt.subplots()
        ax2.bar(hate_result.probas.keys(), hate_result.probas.values(), color='salmon')
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Category")
        ax2.set_title("Hate Speech Prediction Scores")
        st.pyplot(fig2)
