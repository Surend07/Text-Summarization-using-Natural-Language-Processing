import streamlit as st
import streamlit.components.v1 as components
import pyperclip
import time
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    # Tokenize words and sentences
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Remove stopwords and non-alphabetic words
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    return words, sentences

def get_word_frequencies(words):
    # Calculate word frequencies
    freq_table = FreqDist(words)
    return freq_table

def tfidf_weights(text):
    # Calculate TF-IDF weights for words
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    weights = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

    return weights

def calculate_sentence_scores(sentences, tfidf_weights):
    sentence_scores = dict()

    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in tfidf_weights:
                if sentence in sentence_scores:
                    sentence_scores[sentence] += tfidf_weights[word]
                else:
                    sentence_scores[sentence] = tfidf_weights[word]

    return sentence_scores


def calculate_threshold(sentence_scores, target_percentage):
    sorted_scores = sorted(sentence_scores.values(), reverse=True)
    cumulative_sum = 0
    total_sum = sum(sorted_scores)

    for score in sorted_scores:
        cumulative_sum += score
        if cumulative_sum / total_sum >= target_percentage:
            break

    threshold = score
    return threshold

def summarize_text(text):
    words, sentences = preprocess_text(text)
    freq_table = get_word_frequencies(words)

    def tfidf_weights(text):
        # Calculate TF-IDF weights for words
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        weights = dict(zip(feature_names, tfidf_matrix.toarray()[0]))

        return weights

    # Calculate TF-IDF weights for words
    tfidf_weights = tfidf_weights(text)

    sentence_scores = calculate_sentence_scores(sentences, tfidf_weights)
    target_percentage = 0.65
    # Set a threshold for sentence inclusion based on TF-IDF weights
    threshold = calculate_threshold(sentence_scores, target_percentage)

    summary = ''
    for sentence in sentences:
        if sentence_scores.get(sentence, 0) > threshold:
            summary += " " + sentence

    return summary
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Text Summarizer and Chatbot",
        page_icon="✍️",
        layout="wide",
    )
    # Split the window into two columns with adjusted widths
    col1, col2 = st.columns([3, 1])  # Adjust the width of col1 as needed

    # Left column for the chatbot
    with col2:
        components.html(
            """
            <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
            <df-messenger
                intent="WELCOME"
                chat-title="Chatbot"
                agent-id="fd44b178-d24a-4f20-9eae-c76b91eed2ba"
                language-code="en"></df-messenger>
            """,
            height=560,
        )

    # Right column for the text summarizer
    with col1:
        st.title("Text Summarizer")

        # Add a text input area for the user to paste text
        input_text = st.text_area("Paste text here", height=200)

        # Initialize summary outside the button block
        summary = "lol"

        # Add a button to generate the summary
        if st.button("Generate"):
            summary = summarize_text(input_text)
            st.write("Summarized Text:")
            st.write(summary)

        # Add a button to copy the summary to the clipboard
        if st.button("Copy"):

            summary = summarize_text(input_text)
            pyperclip.copy(summary)
            alert = st.success("📋 Summary copied to clipboard!")
            time.sleep(0.3)
            alert.empty()

if __name__ == "__main__":
    main()
