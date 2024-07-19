import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from transformers import pipeline
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import numpy as np

# Function to record and recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success("Recognized Text: " + text)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio")
        except sr.RequestError:
            st.error("Could not request results; check your network connection")
        return ""

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

if st.button("Record Voice"):
    text = recognize_speech()
    if text:
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

        # Topic Context
        topics = topic_modeling(text)
        st.write("Topic Context:")
        st.write(topics)
