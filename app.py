import streamlit as st
import os
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
from annotated_text import annotated_text


# Download NLTK data (run once)
@st.cache_data(show_spinner=False)
def download_nltk_data():
    nltk.download("stopwords")
    nltk.download("punkt")


download_nltk_data()

# Constants
UPLOAD_DIR = "documents"
USER_UPLOADS_DIR = os.path.join(UPLOAD_DIR, "user_uploads")
EXAMPLE_DIR = os.path.join(UPLOAD_DIR, "IrishWomenWWII")

# Ensure the directory exists
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)
os.makedirs(EXAMPLE_DIR, exist_ok=True)

# Supported languages and their corresponding stopwords in NLTK
SUPPORTED_LANGUAGES = {
    "English": "english",
    "German": "german",
    "French": "french",
}

# Load stopwords
STOPWORDS = set(stopwords.words("english"))


# Helper function to read files
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Function to get word frequency with optional stopwords removal
def get_word_frequency(
    text, language="english", remove_stopwords=False, custom_stopwords=None
):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words(language)) if remove_stopwords else set()

    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)

    filtered_words = [
        word for word in words if word.isalnum() and word not in stop_words
    ]
    word_counts = Counter(filtered_words)
    return pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(
        by="Frequency", ascending=False
    )


# Helper function to list files in subfolders
def list_files_in_subfolders(root_dir):
    file_paths = []
    for folder_name, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                relative_path = os.path.relpath(
                    os.path.join(folder_name, filename), root_dir
                )
                file_paths.append(relative_path)
    return sorted(file_paths, key=str.casefold)


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        sentiment_label = "Positive"
    elif sentiment_score < 0:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    return sentiment_score, sentiment_label


def sentence_level_sentiment(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    results = [
        {"Sentence": str(s), "Polarity": s.sentiment.polarity} for s in sentences
    ]
    return pd.DataFrame(results)


# Load spaCy model for NER
@st.cache_resource
def load_spacy_model(language="en"):
    models = {
        "english": "en_core_web_sm",
        "german": "de_core_news_sm",
        "french": "fr_core_news_sm",
    }
    return spacy.load(models[language])


def extract_entities(text, nlp_model):
    doc = nlp_model(text)
    entities = [
        {
            "Text": ent.text,
            "Type": ent.label_,
            "Start": ent.start_char,
            "End": ent.end_char,
        }
        for ent in doc.ents
    ]
    return entities


def main():
    # Page Config
    st.set_page_config(
        page_title="Interview Analysis",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.zhaw.ch/shmr/interview-analysis/blob/main/README.md",
            "Report a bug": "mailto:shmr@zhaw.ch",
            "About": """
# Interview Analysis 🎤
This app is designed to assist researchers, linguists, and professionals in analyzing interview texts. The app allows users to upload and explore interview data, providing tools for text analysis, including word frequency statistics and language-specific stopword filtering.

## Features
- **Upload Your Files**: Upload `.txt` files and analyze them directly in the app.
- **Multilingual Support**: Analyze texts in English, German, or French.
- **Word Frequency Analysis**: Identify the most frequently used words in your interviews with visual charts.
- **Full Text Display**: View and scroll through the complete document in a dedicated tab.

## How to Use
1. Upload a `.txt` file or select from the available documents.
2. Choose the analysis language (English, German, or French).
3. Explore various tabs for insights:
   - **Full Text**: Read the entire document.
   - **Word Frequency**: Analyze word usage and filter stopwords dynamically.

## Contact
- **Help & Documentation**: [Documentation](https://github.zhaw.ch/shmr/interview-analysis/blob/main/README.md)
- **Report Issues**: Please email [shmr@zhaw.ch](mailto:shmr@zhaw.ch) for support or bug reports.

## Credits
Developed by Lars Schmid, research assistant at ZHAW Centre for Artificial Intelligence. Powered by [Streamlit](https://streamlit.io), [NLTK](https://www.nltk.org), and open-source technologies.
""",
        },
    )
    # Streamlit App
    st.title("Interview Analysis")

    # Sidebar to select or upload files
    st.sidebar.header("File Selection")
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

    # Save uploaded file
    if uploaded_file:
        file_path = os.path.join(USER_UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # List available files
    available_files = list_files_in_subfolders(UPLOAD_DIR)
    selected_file = st.sidebar.selectbox(
        "Choose a document", ["None"] + available_files
    )

    # Display basic overview of the selected text
    if selected_file != "None":
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        text = read_txt_file(file_path)

        st.header("Text Overview")
        st.write(f"**Selected File:** {selected_file}")
        st.write(f"**Word Count:** {len(text.split())}")
        st.write(f"**Sentence Count:** {text.count('.')}")

        # Tabs for analysis
        (
            tab_full_text,
            tab_word_frequency,
            tab_sentiment_analysis,
            tab_named_entity_recognition,
        ) = st.tabs(
            [
                "Full Text",
                "Word Frequency",
                "Sentiment Analysis",
                "Named Entity Recognition",
            ]
        )

        # Full Text Tab
        with tab_full_text:
            st.subheader("Full Text")
            st.markdown(text)

        # Inside the Word Frequency Tab
        with tab_word_frequency:
            st.subheader("Word Frequency")

            # Checkbox to enable/disable stopword removal
            remove_stopwords = st.checkbox("Remove Stopwords", value=True)

            # Display additional options only if stopword removal is enabled
            if remove_stopwords:
                st.subheader("Stopword Settings")

                # Language selection
                selected_language = st.selectbox(
                    "Select the language for stopword removal",
                    list(SUPPORTED_LANGUAGES.keys()),
                    index=0,
                )
                language_code = SUPPORTED_LANGUAGES[selected_language]

                # Custom stopwords input
                custom_stopwords_input = st.text_area(
                    "Add custom stopwords (comma-separated)", ""
                )
                custom_stopwords = (
                    set(
                        custom_stopword.strip()
                        for custom_stopword in custom_stopwords_input.split(",")
                    )
                    if custom_stopwords_input
                    else None
                )
            else:
                # Default to no stopwords and language
                language_code = None
                custom_stopwords = None

            # Compute word frequency
            word_freq_df = get_word_frequency(
                text,
                language=language_code,
                remove_stopwords=remove_stopwords,
                custom_stopwords=custom_stopwords,
            )

            (
                tab_word_frequency_bar_chart,
                tab_word_frequency_word_cloud,
                tab_word_frequency_table,
            ) = st.tabs(["Bar Chart", "Word Cloud", "Table"])

            with tab_word_frequency_bar_chart:
                # Allow user to specify the number of words to display
                num_words = st.slider(
                    "Number of Words to Display", min_value=5, max_value=50, value=10
                )
                # Generate and display enhanced bar chart
                top_words = word_freq_df.head(num_words)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_words["Word"], top_words["Frequency"], color="skyblue")
                ax.set_xlabel("Words", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"Top {num_words} Words", fontsize=16)

                # Ensure tick positions match the number of bars
                ax.set_xticks(range(len(top_words["Word"])))
                ax.set_xticklabels(
                    top_words["Word"], rotation=45, ha="right", fontsize=10
                )

                for i, v in enumerate(top_words["Frequency"]):
                    ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
                st.pyplot(fig)

            with tab_word_frequency_word_cloud:
                # Generate and display a word cloud
                st.subheader("Word Cloud")
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate_from_frequencies(
                    dict(zip(word_freq_df["Word"], word_freq_df["Frequency"]))
                )
                st.image(wordcloud.to_array(), use_container_width=True)

            with tab_word_frequency_table:
                # Display data table
                st.dataframe(
                    data=word_freq_df,
                    hide_index=True,
                    column_config={
                        "Word": st.column_config.TextColumn("Word"),
                        "Frequency": st.column_config.NumberColumn("Frequency"),
                    },
                )

        # Sentiment Analysis Tab
        with tab_sentiment_analysis:
            st.subheader("Sentiment Analysis")

            # Overall Sentiment
            sentiment_score, sentiment_label = analyze_sentiment(text)
            st.metric(
                "Overall Sentiment", sentiment_label, f"Polarity: {sentiment_score:.2f}"
            )

            tab_sentiment_analysis_sentences, tab_sentiment_analysis_distribution = (
                st.tabs(["Sentence-Level Sentiment", "Sentiment Polarity Distribution"])
            )

            with tab_sentiment_analysis_sentences:
                # Sentence-Level Sentiment
                st.subheader("Sentence-Level Sentiment")
                sentence_sentiments = sentence_level_sentiment(text)
                st.dataframe(sentence_sentiments)

            with tab_sentiment_analysis_distribution:
                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sentence_sentiments["Polarity"].hist(
                    bins=20, ax=ax, color="skyblue", edgecolor="black"
                )
                ax.set_title("Sentiment Polarity Distribution", fontsize=16)
                ax.set_xlabel("Polarity", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                st.pyplot(fig)

        # NER Tab
        with tab_named_entity_recognition:
            st.subheader("Named Entity Recognition")

            # Language selection for NER
            ner_language = st.selectbox(
                "Select language for NER",
                list(SUPPORTED_LANGUAGES.keys()),
                index=0,
            )
            nlp = load_spacy_model(SUPPORTED_LANGUAGES[ner_language])

            # Extract entities
            entities = extract_entities(text, nlp)
            entity_df = pd.DataFrame(entities)
            available_entity_types = sorted(
                entity_df["Type"].unique(), key=str.casefold
            )

            # Get unique entity types and explanations
            entity_explanations = {
                entity: spacy.explain(entity) or "No description available"
                for entity in available_entity_types
            }

            # Add explanations as a helper text
            with st.expander("Show/hide explanations for entity types", expanded=True):
                for entity, explanation in entity_explanations.items():
                    st.caption(f"**{entity}**: {explanation}")

            (
                tab_named_entity_recognition_highlighting,
                tab_named_entity_recognition_table,
                tab_named_entity_recognition_bar_chart,
            ) = st.tabs(["Entity Highlighting", "Table", "Bar Chart"])

            with tab_named_entity_recognition_highlighting:
                st.subheader("Entity Highlighting")
                if not entity_df.empty:
                    # Default to showing all entities highlighted
                    default_selected_types = available_entity_types

                    # Form for entity type selection
                    with st.form("entity_type_selection"):
                        selected_entity_types = st.multiselect(
                            "Select entity types to highlight:",
                            options=available_entity_types,
                            default=default_selected_types,  # Default to all entity types selected
                        )
                        # Submit button
                        submitted = st.form_submit_button("Apply")

                    # Filter entities based on selected types
                    filtered_entities = (
                        [
                            ent
                            for ent in entities
                            if ent["Type"] in selected_entity_types
                        ]
                        if submitted
                        else entities
                    )

                    # Annotate text using `annotated_text`
                    annotated_segments = []
                    last_end = 0
                    for ent in filtered_entities:
                        # Add plain text before the entity
                        if ent["Start"] > last_end:
                            annotated_segments.append(text[last_end : ent["Start"]])
                        # Add the highlighted entity
                        annotated_segments.append((ent["Text"], ent["Type"]))
                        last_end = ent["End"]

                    # Add remaining plain text after the last entity
                    if last_end < len(text):
                        annotated_segments.append(text[last_end:])

                    # Display annotated text
                    if annotated_segments:
                        annotated_text(*annotated_segments)
                    else:
                        st.warning("No entities of the selected types found.")
                else:
                    st.write("No entities found in the document.")

            with tab_named_entity_recognition_table:
                # Display entities table
                if not entity_df.empty:
                    st.dataframe(
                        data=entity_df,
                        hide_index=True,
                        column_config={
                            "Text": st.column_config.TextColumn("Text"),
                            "Type": st.column_config.TextColumn("Type"),
                            "Start": st.column_config.NumberColumn("Start"),
                            "End": st.column_config.NumberColumn("End"),
                        },
                    )
                    entity_counts = entity_df["Type"].value_counts().reset_index()
                    entity_counts.columns = ["Entity Type", "Count"]
                else:
                    st.write("No entities found in the document.")

            with tab_named_entity_recognition_bar_chart:
                if not entity_df.empty:
                    # Generate enhanced bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(
                        entity_counts["Entity Type"],
                        entity_counts["Count"],
                        color="skyblue",
                    )
                    ax.set_xlabel("Entity Types", fontsize=12)
                    ax.set_ylabel("Count", fontsize=12)
                    ax.set_title("Entity Type Distribution", fontsize=16)

                    # Ensure tick positions match the number of bars
                    ax.set_xticks(range(len(entity_counts["Entity Type"])))
                    ax.set_xticklabels(
                        entity_counts["Entity Type"],
                        rotation=45,
                        ha="right",
                        fontsize=10,
                    )

                    # Add text annotations above the bars
                    for i, v in enumerate(entity_counts["Count"]):
                        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)

                    st.pyplot(fig)
                else:
                    st.write("No entities found in the document.")
    else:
        st.warning("No document selected")


if __name__ == "__main__":
    main()
