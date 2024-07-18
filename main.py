import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from PIL import Image
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import contractions

# Load Spacy model
nlp = spacy.load('en_core_web_sm')


# News API Key
api_key = '7a3ff37653f347c5a234987df2fecfd7'

# Fetch and clean news data
def fetch_news(query, language='en', page_size=100):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    news_data = response.json()
    return news_data

# clean text function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    tokens = [token.lemma_ for token in nlp(' '.join(tokens))]
    # Remove extra whitespace
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Generated LDA model using sklearn with vocabulary filtering
def generate_lda(cleaned_articles):
    vectorizer = CountVectorizer(max_df=0.75, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(cleaned_articles)
    
    # Get feature names and filter unwanted words
    vocab = vectorizer.get_feature_names_out()
    unwanted_words = {'char', 'chars'}  
    filtered_vocab = [word for word in vocab if word not in unwanted_words]
    
    # Update the CountVectorizer with the filtered vocabulary
    vectorizer = CountVectorizer(vocabulary=filtered_vocab)
    doc_term_matrix = vectorizer.fit_transform(cleaned_articles)
    
    lda_model = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online')
    lda_model.fit(doc_term_matrix)
    
    return lda_model, vectorizer

# Generate word cloud
def generate_wordcloud(lda_model, vectorizer):
    words = vectorizer.get_feature_names_out()
    topics = lda_model.components_
    
    topic_words = []
    for topic in topics:
        topic_words.append([words[i] for i in topic.argsort()[:-11:-1]])
    
    all_topics = ' '.join([' '.join(topic) for topic in topic_words])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_topics)
    return wordcloud

# Generate bar diagram for top-5 topics
def generate_bar_diagram(lda_model, vectorizer, num_topics=5):
    words = vectorizer.get_feature_names_out()
    topics = lda_model.components_

    topic_summaries = []
    for topic in topics:
        topic_summary = {words[i]: topic[i] for i in topic.argsort()[:-11:-1]}
        topic_summaries.append(topic_summary)
    
    topic_df = pd.DataFrame(topic_summaries)
    top_words = topic_df.sum().sort_values(ascending=False).head(num_topics)
    
    plt.figure(figsize=(10, 5))
    plt.barh(top_words.index, top_words.values, color='skyblue')
    plt.xlabel('Word Frequency')
    plt.title('Top-5 Topics')
    st.pyplot(plt)

# Streamlit App
st.title('📰 News Topic Modeling and Word Cloud')
st.markdown("## Step 1: Enter a Topic to Search for News")

# Input field for the news topic
query = st.text_input('Enter a topic to search for news:', 'technology')

if st.button('Fetch News and Generate Word Cloud'):
    st.markdown("## Step 2: Fetching News and Cleaning Data")
    
    # Fetching news articles based on the query
    news_data = fetch_news(query)
    articles = [article['content'] for article in news_data['articles'] if article['content']]
    cleaned_articles = [clean_text(article) for article in articles]
    
    st.markdown("### News Articles DataFrame")
    
    # Displaying the news  articles DataFrame
    articles_df = pd.DataFrame(news_data['articles'])
    st.dataframe(articles_df.head())
    
    st.markdown("### Cleaned Articles DataFrame ")
    
    # Displaying cleaned news articles DataFrame
    cleaned_articles_df = pd.DataFrame({'cleaned_text': cleaned_articles})
    st.dataframe(cleaned_articles_df.head())
    
    st.markdown("## Step 3: Generating LDA Model and Word Cloud")
    
    # Generate LDA model and word cloud
    lda_model, vectorizer = generate_lda(cleaned_articles)
    wordcloud = generate_wordcloud(lda_model, vectorizer)
    
    # Display the word cloud
    st.image(wordcloud.to_array())
    
    st.markdown("## Step 4: Top-5 Topics")
    
    # Generate and display the bar diagram for top-5 topics
    generate_bar_diagram(lda_model, vectorizer, num_topics=5)
