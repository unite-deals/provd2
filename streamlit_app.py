import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from textblob import TextBlob
from transformers import pipeline
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment

# Function to classify text
@st.cache(allow_output_mutation=True)
def load_classifier():
    return pipeline("zero-shot-classification")

def classify_text(text, classifier):
    labels = ["business", "technology", "entertainment", "sports", "politics"]
    result = classifier(text, candidate_labels=labels)
    return result

# Function to determine topic context
def topic_modeling(text):
    words = text.lower().split()
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    topics = lda_model.print_topics(num_words=4)
    return topics

# Streamlit interface
st.title("Live Speech-to-Text with Sentiment Analysis, Text Classification, and Topic Modeling")

state = st.session_state

if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Convert speech to text:")
with c2:
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

if text:
    state.text_received.append(text)

for text in state.text_received:
    st.text(text)

    # Sentiment Analysis
    sentiment = analyze_sentiment(text)
    st.write("Sentiment Analysis:")
    st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

    # Text Classification
    classifier = load_classifier()
    classification = classify_text(text, classifier)
    st.write("Text Classification:")
    st.write(classification)

    # Topic Modeling
    topics = topic_modeling(text)
    st.write("Topic Context:")
    for topic in topics:
        st.write(topic)

st.write("Record your voice, and play the recorded audio:")
audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

if audio:
    st.audio(audio['bytes'])
