# Interview Analysis 📝

This app is designed to provide assistance for analyzing interview or text data. It provides various tools for in-depth text analysis, including word frequency, sentiment analysis, and named entity recognition (NER), with support for multiple languages.

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

## Installation

This section provides detailed steps to install and run the app in your local environment. You can create a new Python environment using pyenv, conda, or python venv. Once the environment is set up, install the dependencies and launch the app.

### Step 1: Clone the Repository

Open your terminal and navigate to the directory where you want to install the app. Then, clone the repository by running:

```
git clone https://github.com/larscarl/interview-analysis.git
cd  interview-analysis
```

### Step 2: Create a Python Environment

Choose one of the following methods to create a new Python environment. In this example, we use Python 3.10.15, but 3.9 works as well.

#### A. Using `pyenv`

1. Install pyenv (if not already installed)
2. Install Python 3.10.15 (if not already installed):

```
pyenv install 3.10.15
```

3. Create a virtual environment:

```
pyenv virtualenv 3.10.15 interview-analysis
```

#### B. Using conda

1. Install Conda (if not already installed)
2. Create a Conda environment:

```
conda create -n interview-analysis python=3.10.15 -y
conda activate interview-analysis
```

#### C. Using python venv

1. Ensure Python 3.10.15 is installed: Check your Python version:

```
python --version
```

2. Create a virtual environment:

```
python -m venv venv
```

3. Activate the environment:

- On macOS/Linux:

```
source venv/bin/activate
```

- On Windows:

```
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the required dependencies using the `requirements.txt` file:

```
pip install -r requirements.txt
```

This command installs all necessary libraries and models.

### Step 4: Run the Application

Once dependencies are installed, launch the app:

```
streamlit run app.py
```

This will start a local web server and provide a URL (e.g., http://localhost:8501) to access the app in your browser.

### Additional Notes

- Installing `spaCy` Language Models: The `requirements.txt` file includes direct links to `spaCy` language models (e.g., `en_core_web_sm`, `de_core_news_sm`, `fr_core_news_sm`). They will be downloaded automatically during the dependency installation.
- Example Files: The app includes a folder called "IrishWomenWWII" under the documents directory for testing. You can upload additional `.txt` files using the file upload feature in the app. They will be stored in the directory `documents/user_uploads`.
- Environment Verification: Ensure all dependencies are installed correctly by running:

```
pip list
```

- Deactivate the Environment: When you're done, deactivate your environment:

```
# Conda:
conda deactivate

# python venv:
deactivate

# pyenv:
pyenv deactivate
```

## How to Use the App

1. Upload a `.txt` file or select from the available example documents.
2. Choose the analysis options you need:
   - View the full text.
   - Explore word frequency data with dynamic charts and tables.
   - Perform sentiment analysis and interact with visualizations.
   - Extract and analyze named entities interactively.
3. Adjust settings dynamically (e.g., stopword filters, smoothing options, or entity type selection).

## Contact

- **Help & Documentation**: [Documentation](https://github.com/larscarl/interview-analysis/blob/main/README.md)
- **Report Issues**: Please email [shmr@zhaw.ch](mailto:shmr@zhaw.ch) for support or bug reports.

## Credits

Developed by [Lars Schmid](mailto:shmr@zhaw.ch), research assistant at [ZHAW Centre for Artificial Intelligence](https://www.zhaw.ch/en/engineering/institutes-centres/cai/). Built with:

- [Streamlit](https://streamlit.io) for the interactive web app.
- [NLTK](https://www.nltk.org) and [TextBlob](https://textblob.readthedocs.io/en/dev/) for text analysis.
- [spaCy](https://spacy.io) for advanced NLP and named entity recognition.
- [Plotly](https://plotly.com) for interactive visualizations.
- [WordCloud](https://github.com/amueller/word_cloud) for word cloud generation.
