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
import plotly.graph_objects as go
import re


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


def main():
    download_nltk_data()
    set_up_streamlit_app()
    user_uploads_file()
    selected_file = user_selects_file()
    if selected_file != "None":
        text = load_text_from_selection(selected_file)
        text = normalize_line_breaks(text)

        show_text_overview(selected_file, text)

        (
            tab_full_text,
            tab_word_frequency,
            tab_sentiment_analysis,
            tab_named_entity_recognition,
        ) = create_tabs_text_analysis()
        show_tab_full_text(text, tab_full_text)
        show_tab_word_frequency(text, tab_word_frequency)
        show_tab_sentiment_analysis(text, tab_sentiment_analysis)
        show_tab_named_entity_recognition(text, tab_named_entity_recognition)
    else:
        st.warning("No document selected")


def set_up_streamlit_app():
    st.set_page_config(
        page_title="Interview Analysis",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.zhaw.ch/shmr/interview-analysis/blob/main/README.md",
            "Report a bug": "mailto:shmr@zhaw.ch",
            "About": """
# Interview Analysis 📝
This app is designed to assist researchers, linguists, and professionals in analyzing interview or text data. It provides various tools for in-depth text analysis, including word frequency, sentiment analysis, and named entity recognition (NER), with support for multiple languages.
## Features
- **Upload Your Files**: Upload `.txt` files and analyze them directly in the app.
- **Text Display and Normalization**: View the entire text with paragraph normalization for better readability.
- **Word Frequency Analysis**: Identify the most frequently used words in your text, with options to filter stopwords and add custom stopwords.
- **Word Cloud Visualization**: Generate word clouds for a quick visual representation of word usage.
- **Sentiment Analysis**:
  - **Overall Sentiment**: Get the overall sentiment of the document.
  - **Sentence-Level Analysis**: Explore the sentiment polarity for each sentence in a tabular format.
  - **Polarity Timeline (Interactive)**: Visualize sentiment changes over the text with an interactive scatter plot and smoothed trendlines.
  - **Sentiment Distribution**: View the polarity distribution using histograms.
- **Named Entity Recognition (NER)**:
  - Extract entities like names, organizations, locations, and more using advanced NLP techniques.
  - Highlight entities directly in the text with interactive filters for entity types.
  - Display detailed tables of extracted entities and their occurrences.
  - View entity type distributions with bar charts.
- **Multilingual Support**: Analyze texts in English, German, or French.
## How to Use
1. Upload a `.txt` file or select from the available example documents.
2. Choose the analysis options you need:
   - View the full text.
   - Explore word frequency data with dynamic charts and tables.
   - Perform sentiment analysis and interact with visualizations.
   - Extract and analyze named entities interactively.
3. Adjust settings dynamically (e.g., stopword filters, smoothing options, or entity type selection).
## Contact
- **Help & Documentation**: [Documentation](https://github.zhaw.ch/shmr/interview-analysis/blob/main/README.md)
- **Report Issues**: Please email [shmr@zhaw.ch](mailto:shmr@zhaw.ch) for support or bug reports.
## Credits
Developed by Lars Schmid, research assistant at ZHAW Centre for Artificial Intelligence. Built with:
- [Streamlit](https://streamlit.io) for the interactive web app.
- [NLTK](https://www.nltk.org) and [TextBlob](https://textblob.readthedocs.io/en/dev/) for text analysis.
- [spaCy](https://spacy.io) for advanced NLP and named entity recognition.
- [Plotly](https://plotly.com) for interactive visualizations.
- [WordCloud](https://github.com/amueller/word_cloud) for word cloud generation.
""",
        },
    )
    st.title("Interview Analysis")


def user_uploads_file():
    st.sidebar.header("File Selection")
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

    # Save uploaded file
    if uploaded_file:
        file_path = os.path.join(USER_UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")


def user_selects_file():
    # List available files
    available_files = list_files_in_subfolders(UPLOAD_DIR)
    selected_file = st.sidebar.selectbox(
        "Choose a document", ["None"] + available_files
    )

    return selected_file


def list_files_in_subfolders(root_dir):
    # Helper function to list files in subfolders
    file_paths = []
    for folder_name, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                relative_path = os.path.relpath(
                    os.path.join(folder_name, filename), root_dir
                )
                file_paths.append(relative_path)
    return sorted(file_paths, key=str.casefold)


def load_text_from_selection(selected_file):
    file_path = os.path.join(UPLOAD_DIR, selected_file)
    text = read_txt_file(file_path)

    return text


def normalize_line_breaks(text):
    # Normalize line breaks: convert single newlines to double newlines for paragraph separation,
    # and collapse excessive newlines (more than two) into a maximum of two.
    text = re.sub(r"\n+", "\n\n", text)
    return text


def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def show_text_overview(selected_file, text):
    st.header("Text Overview")
    st.write(f"**Selected File:** {selected_file}")
    st.write(f"**Word Count:** {len(text.split())}")
    st.write(f"**Sentence Count:** {text.count('.')}")


def create_tabs_text_analysis():
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

    return (
        tab_full_text,
        tab_word_frequency,
        tab_sentiment_analysis,
        tab_named_entity_recognition,
    )


def show_tab_full_text(text, tab_full_text):
    with tab_full_text:
        st.text(text)


def show_tab_word_frequency(text, tab_word_frequency):
    with tab_word_frequency:
        col1, col2 = st.columns([0.2, 0.8])
        # Checkbox to enable/disable stopword removal
        remove_stopwords = col1.checkbox("Remove Stopwords", value=True)
        selected_language = user_selects_language_for_stopword_removal(col2)
        language_code = SUPPORTED_LANGUAGES[selected_language]

        custom_stopwords_input = user_selects_custom_stopwords(col2)
        custom_stopwords = (
            generate_stopwords_from_selection(custom_stopwords_input)
            if custom_stopwords_input
            else None
        )

        if not remove_stopwords:
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
        ) = create_tabs_word_frequency()
        show_tab_word_frequency_bar_chart(word_freq_df, tab_word_frequency_bar_chart)
        show_tab_word_frequency_word_cloud(word_freq_df, tab_word_frequency_word_cloud)
        show_tab_word_frequency_table(word_freq_df, tab_word_frequency_table)


def user_selects_language_for_stopword_removal(col2):
    selected_language = col2.selectbox(
        "Select the language for stopword removal",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0,
    )

    return selected_language


def generate_stopwords_from_selection(custom_stopwords_input):
    return set(
        custom_stopword.strip() for custom_stopword in custom_stopwords_input.split(",")
    )


def user_selects_custom_stopwords(col2):
    custom_stopwords_input = col2.text_area(
        "Add custom stopwords (comma-separated)", ""
    )

    return custom_stopwords_input


def get_word_frequency(
    text, language="english", remove_stopwords=False, custom_stopwords=None
):
    # Function to get word frequency with optional stopwords removal
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words(language)) if remove_stopwords else set()

    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)

    filtered_words = [
        word for word in words if word.isalnum() and word not in stop_words
    ]
    word_counts = Counter(filtered_words)
    word_frequency_df = pd.DataFrame(
        word_counts.items(), columns=["Word", "Frequency"]
    ).sort_values(by="Frequency", ascending=False)
    return word_frequency_df


def create_tabs_word_frequency():
    (
        tab_word_frequency_bar_chart,
        tab_word_frequency_word_cloud,
        tab_word_frequency_table,
    ) = st.tabs(["Bar Chart", "Word Cloud", "Table"])

    return (
        tab_word_frequency_bar_chart,
        tab_word_frequency_word_cloud,
        tab_word_frequency_table,
    )


def show_tab_word_frequency_bar_chart(word_freq_df, tab_word_frequency_bar_chart):
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
        ax.set_xticklabels(top_words["Word"], rotation=45, ha="right", fontsize=10)

        for i, v in enumerate(top_words["Frequency"]):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)
        st.pyplot(fig)


def show_tab_word_frequency_word_cloud(word_freq_df, tab_word_frequency_word_cloud):
    with tab_word_frequency_word_cloud:
        # Generate and display a word cloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(
            dict(zip(word_freq_df["Word"], word_freq_df["Frequency"]))
        )
        st.image(wordcloud.to_array(), use_container_width=True)


def show_tab_word_frequency_table(word_freq_df, tab_word_frequency_table):
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


def show_tab_sentiment_analysis(text, tab_sentiment_analysis):
    with tab_sentiment_analysis:
        # Overall Sentiment
        sentiment_score, sentiment_label = analyze_sentiment(text)
        st.metric(
            "Overall Sentiment", sentiment_label, f"Polarity: {sentiment_score:.2f}"
        )
        (
            tab_sentiment_analysis_sentences,
            tab_sentiment_analysis_timeline,
            tab_sentiment_analysis_distribution,
        ) = create_tabs_sentiment_analysis()
        sentence_sentiments = generate_sentence_level_sentiments(text)
        show_tab_sentiment_analysis_sentences(
            sentence_sentiments, tab_sentiment_analysis_sentences
        )
        show_tab_sentiment_analysis_timeline(
            sentence_sentiments, tab_sentiment_analysis_timeline
        )
        show_tab_sentiment_analysis_distribution(
            sentence_sentiments, tab_sentiment_analysis_distribution
        )


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


def create_tabs_sentiment_analysis():
    (
        tab_sentiment_analysis_sentences,
        tab_sentiment_analysis_timeline,
        tab_sentiment_analysis_distribution,
    ) = st.tabs(
        [
            "Sentence-Level Sentiment",
            "Polarity Timeline (Interactive)",
            "Sentiment Polarity Distribution",
        ]
    )

    return (
        tab_sentiment_analysis_sentences,
        tab_sentiment_analysis_timeline,
        tab_sentiment_analysis_distribution,
    )


def generate_sentence_level_sentiments(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    results = [
        {"Sentence": str(s), "Polarity": s.sentiment.polarity} for s in sentences
    ]
    return pd.DataFrame(results)


def show_tab_sentiment_analysis_sentences(
    sentence_sentiments, tab_sentiment_analysis_sentences
):
    with tab_sentiment_analysis_sentences:
        # Sentence-Level Sentiment
        st.dataframe(
            data=sentence_sentiments,
            hide_index=False,
            column_config={
                "Sentence": st.column_config.TextColumn("Sentence"),
                "Polarity": st.column_config.NumberColumn("Polarity", step=0.01),
            },
        )

    return sentence_sentiments


def show_tab_sentiment_analysis_timeline(
    sentence_sentiments, tab_sentiment_analysis_timeline
):
    with tab_sentiment_analysis_timeline:
        # Extract polarity values
        sentence_sentiments["Position"] = range(1, len(sentence_sentiments) + 1)

        # User option for smoothing
        smoothing_window = st.slider(
            "Smoothing Window (Number of Sentences)",
            min_value=1,
            max_value=min(10, len(sentence_sentiments)),
            value=3,
            step=1,
        )

        # Compute moving average for polarity
        sentence_sentiments["Smoothed_Polarity"] = (
            sentence_sentiments["Polarity"]
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
        )

        # Create an interactive scatter plot with hover tooltips
        fig = go.Figure()

        # Add scatter points for original polarity
        fig.add_trace(
            go.Scatter(
                x=sentence_sentiments["Position"],
                y=sentence_sentiments["Polarity"],
                mode="markers",
                marker=dict(
                    color=sentence_sentiments["Polarity"],
                    colorscale="RdYlGn",
                    showscale=True,
                    size=8,
                    line=dict(color="black", width=0.5),
                    colorbar=dict(
                        title="Polarity",
                        titleside="right",
                        x=1.03,  # Position closer to the plot
                        y=0.5,
                        thickness=20,  # Adjust thickness
                    ),
                ),
                name="Original Polarity",
                hovertemplate=(
                    "<b>Sentence:</b> %{customdata[0]}<br>"
                    "<b>Polarity:</b> %{y:.2f}<br>"
                    "<b>Position:</b> %{x}<extra></extra>"
                ),
                customdata=[
                    [sentence] for sentence in sentence_sentiments["Sentence"].to_list()
                ],
            )
        )

        # Add smoothed polarity as a line
        fig.add_trace(
            go.Scatter(
                x=sentence_sentiments["Position"],
                y=sentence_sentiments["Smoothed_Polarity"],
                mode="lines",
                line=dict(color="blue", width=2),
                name="Smoothed Polarity",
            )
        )

        # Customize layout
        fig.update_layout(
            xaxis=dict(
                title="Sentence Position",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                range=[
                    0,
                    len(sentence_sentiments["Position"]),
                ],  # Ensure x-axis starts at 0
            ),
            yaxis=dict(
                title="Polarity",
                title_font=dict(size=14),
                tickfont=dict(size=12),
                gridcolor="lightgray",  # Horizontal grid lines
                gridwidth=1,  # Thickness of grid lines
                tickvals=[-1, -0.5, 0, 0.5, 1],  # Custom tick positions
                ticktext=["-1", "-0.5", "0", "0.5", "1"],  # Tick labels
                showgrid=True,  # Explicitly enable gridlines
            ),
            template="simple_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )

        # Render plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_tab_sentiment_analysis_distribution(
    sentence_sentiments, tab_sentiment_analysis_distribution
):
    with tab_sentiment_analysis_distribution:
        # Sentiment Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        sentence_sentiments["Polarity"].hist(
            bins=20, ax=ax, color="skyblue", edgecolor="black"
        )
        ax.set_title("Sentiment Polarity Distribution", fontsize=16)
        ax.set_xlabel("Polarity", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        st.pyplot(fig)


def show_tab_named_entity_recognition(text, tab_ner):
    with tab_ner:
        # Language selection for NER
        ner_language = user_selects_language_for_ner()
        nlp = load_spacy_model(SUPPORTED_LANGUAGES[ner_language])

        entities = extract_entities(text, nlp)
        entity_df = pd.DataFrame(entities)

        custom_exclusions = user_selects_custom_exclusions()

        if custom_exclusions:
            entity_df = filter_excluded_entities(entities, custom_exclusions)
        else:
            filtered_entities = entities  # No exclusions applied

        if not entity_df.empty:
            available_entity_types = get_entity_types(entity_df)

            # Get unique entity types and explanations
            explanations = get_entity_type_explanations(available_entity_types)

            # Add explanations as a helper text
            show_or_hide_entity_explanations(explanations)

            (
                tab_ner_highlighting,
                tab_ner_table,
                tab_ner_bar_chart,
            ) = st.tabs(["Entity Highlighting", "Table", "Bar Chart"])

            show_tab_ner_highlighting(
                text,
                entity_df,
                filtered_entities,
                available_entity_types,
                tab_ner_highlighting,
            )
            show_tab_ner_table(entity_df, tab_ner_table)
            show_tab_ner_bar_chart(entity_df, tab_ner_bar_chart)
        else:
            st.warning("No entities available after applying exclusions.")


def user_selects_language_for_ner():
    ner_language = st.selectbox(
        "Select language",
        list(SUPPORTED_LANGUAGES.keys()),
        index=0,
    )

    return ner_language


@st.cache_resource
def load_spacy_model(language="en"):
    # Load spaCy model for NER
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


def user_selects_custom_exclusions():
    custom_exclusions_input = st.text_area(
        "Add custom words to exclude from Named Entity Recognition (comma-separated)",
        "",
    )
    custom_exclusions = [
        word.strip() for word in custom_exclusions_input.split(",") if word.strip()
    ]

    return custom_exclusions


def filter_excluded_entities(entities, custom_exclusions):
    filtered_entities = [
        ent
        for ent in entities
        if ent["Text"].lower() not in map(str.lower, custom_exclusions)
    ]
    entity_df = pd.DataFrame(filtered_entities)
    return entity_df


def get_entity_types(entity_df):
    entity_types = sorted(entity_df["Type"].unique(), key=str.casefold)

    return entity_types


def get_entity_type_explanations(available_entity_types):
    explanations = {
        entity: spacy.explain(entity) or "No description available"
        for entity in available_entity_types
    }

    return explanations


def show_or_hide_entity_explanations(entity_explanations):
    with st.expander("Show/hide explanations for entity types", expanded=True):
        for entity, explanation in entity_explanations.items():
            st.write(f"**{entity}**: {explanation}")


def show_tab_ner_highlighting(
    text,
    entity_df,
    filtered_entities,
    available_entity_types,
    tab_ner_highlighting,
):
    with tab_ner_highlighting:
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
            filtered_entities_by_type = (
                [
                    ent
                    for ent in filtered_entities
                    if ent["Type"] in selected_entity_types
                ]
                if submitted
                else filtered_entities
            )

            annotated_segments = []
            last_end = 0
            for ent in filtered_entities_by_type:
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


def show_tab_ner_table(entity_df, tab_ner_table):
    with tab_ner_table:
        # Display entities table
        if not entity_df.empty:
            st.dataframe(
                data=entity_df,
                hide_index=False,
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


def show_tab_ner_bar_chart(entity_df, tab_ner_bar_chart):
    with tab_ner_bar_chart:
        if not entity_df.empty:
            # Generate enhanced bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            entity_counts = entity_df["Type"].value_counts().reset_index()
            entity_counts.columns = ["Entity Type", "Count"]
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


# Download NLTK data (run once)
@st.cache_data(show_spinner=False)
def download_nltk_data():
    nltk.download("stopwords")
    nltk.download("punkt")


if __name__ == "__main__":
    main()
