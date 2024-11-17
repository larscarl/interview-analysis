import streamlit as st
import os
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK data (run once)
nltk.download("stopwords")
nltk.download("punkt")

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
    selected_file = st.sidebar.selectbox("Choose a document", available_files)

    # Language selection
    st.sidebar.header("Language Selection")
    selected_language = st.sidebar.selectbox(
        "Select the language", list(SUPPORTED_LANGUAGES.keys())
    )
    language_code = SUPPORTED_LANGUAGES[selected_language]

    # Custom stopwords input
    st.sidebar.header("Custom Stopwords")
    custom_stopwords_input = st.sidebar.text_area(
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

    # Display basic overview of the selected text
    if selected_file:
        file_path = os.path.join(UPLOAD_DIR, selected_file)
        text = read_txt_file(file_path)

        st.header("Text Overview")
        st.write(f"**Selected File:** {selected_file}")
        st.write(f"**Word Count:** {len(text.split())}")
        st.write(f"**Sentence Count:** {text.count('.')}")
        st.write("**Preview:**")
        st.text(text[:500] + ("..." if len(text) > 500 else ""))

        # Tabs for analysis
        tab_full_text, tab_word_frequency, tab_placeholder = st.tabs(
            ["Full Text", "Word Frequency", "Other Analyses (Coming Soon)"]
        )

        # Full Text Tab
        with tab_full_text:
            st.subheader("Full Text")
            st.text_area("Full Document", text, height=400)

        # Inside the Word Frequency Tab
        with tab_word_frequency:
            st.subheader("Word Frequency")

            # Add a checkbox to filter stopwords
            remove_stopwords = st.checkbox("Remove Stopwords", value=True)

            # Allow user to specify the number of words to display
            num_words = st.slider(
                "Number of Words to Display", min_value=5, max_value=50, value=10
            )

            # Compute word frequency
            word_freq_df = get_word_frequency(
                text,
                language=language_code,
                remove_stopwords=remove_stopwords,
                custom_stopwords=custom_stopwords,
            )

            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Word Cloud", "Table"])

            with tab1:
                # Generate and display enhanced bar chart
                # st.subheader()
                top_words = word_freq_df.head(num_words)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(top_words["Word"], top_words["Frequency"], color="skyblue")
                ax.set_xlabel("Words", fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"Top {num_words} Words", fontsize=16)
                ax.set_xticklabels(
                    top_words["Word"], rotation=45, ha="right", fontsize=10
                )
                for i, v in enumerate(top_words["Frequency"]):
                    ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
                st.pyplot(fig)

            with tab2:
                # Generate and display a word cloud
                st.subheader("Word Cloud")
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate_from_frequencies(
                    dict(zip(word_freq_df["Word"], word_freq_df["Frequency"]))
                )
                st.image(wordcloud.to_array(), use_container_width=True)

            with tab3:
                # Display data table
                st.dataframe(
                    data=word_freq_df,
                    hide_index=True,
                    column_config={
                        "Word": st.column_config.TextColumn("Word"),
                        "Frequency": st.column_config.NumberColumn("Frequency"),
                    },
                )

        # Placeholder Tab
        with tab_placeholder:
            st.subheader("Analysis Coming Soon")
            st.write("More analysis features will be added here.")


if __name__ == "__main__":
    main()
