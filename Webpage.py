import streamlit as st
import streamlit.components.v1 as components
import pyperclip
import time
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Define a function to summarize text
def summarize_text(text):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence


    return summary


# Embed Dialogflow messenger code

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
