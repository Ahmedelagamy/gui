# -*- coding: utf-8 -*-
"""

@author: Ahmed El Agamy
"""
import pandas as pd
# Imports
import streamlit as st
from bertopic import BERTopic
from textblob import TextBlob
from umap import UMAP
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from langdetect import detect

# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

# Functions for interpreting TextBlob analysis
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity


# Applying Analysis Function
def get_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# Code for project structure
st.image('piovos_logo.png')
st.title("Piovis Automate")
st.sidebar.title('Review analyzer GUI')
st.markdown("This application is a streamlit deployment to automate analysis")


# Loading Data
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_excel(uploaded_file)
  st.write(df)
else:
  st.stop()


# Applying language detection
df.dropna(inplace=True)

text_col = df['review-text'].astype(str)
langdet = []
# Data preprocessing
for i in range(len(df)):
    try:
        lang=detect(text_col[i])
    except:
        lang='no'

    langdet.append(lang)

df['detect'] = langdet

# Select language here
en_df = df[df['detect'] == 'en']


# Applying sentiment analysis
en_df['TextBlob_Polarity'] = en_df['review-text'].astype(str).apply(get_polarity)
en_df['TextBlob_Analysis'] = en_df['TextBlob_Polarity'].apply(get_analysis)


# Splitting data
bad_reviews = en_df[en_df['TextBlob_Analysis'] == 'Negative']
good_reviews = en_df[en_df['TextBlob_Analysis'] == 'Positive']

# Minor mod
st.header('Select Stop Words')

custom_stopwords = st.text_input('Enter Stopword')
cutom_stopwords = custom_stopwords.split()
final_stop_words = set(stopwords.words("english")).union(custom_stopwords)

def clean_text(dataframe, col_name):
    
    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    word = "inversely"
    print("stemming:", stem.stem(word))
    print("lemmatization:", lem.lemmatize(word, "v"))
    stop_words = final_stop_words

    docs = []
    for i in dataframe[col_name]:
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', i)

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        # Convert to list from string
        text = text.split()

        # Stemming
        PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]

        text = " ".join(text)
        # print(text)
        docs.append(text)
    # print(docs)
    return docs


# Applying function
bad_reviews_data = clean_text(bad_reviews, 'review-text')
good_reviews_data=clean_text(good_reviews, 'review-text')
final_df= en_df.groupby(['asin']).mean()
good_topic_info= pd.DataFrame()
bad_topic_info= pd.DataFrame
# Tab Structure
tab = st.sidebar.selectbox('Pick one', ['Positive Review', 'Negative Review'])

# Insert containers separated into tabs:
topic_model = BERTopic(language= 'en', n_gram_range= (2,4), diversity=.5, verbose=True)

# Models
if tab == 'Positive Review':
    
    st.subheader('Positive Reviews')
    st.dataframe(good_reviews)
    
# Fixing small dataset bug
    if len(good_reviews) < 300: # Workaround if not enough documents https://github.com/MaartenGr/BERTopic/issues/97 , https://github.com/MaartenGr/Concept/issues/5
       good_reviews_data.extend(3*good_reviews_data)

    good_model = topic_model.fit(good_reviews_data)
    """# Good Reviews model insight"""

    

    good_model.get_topic_info()
    
    topic_labels = good_model.generate_topic_labels(nr_words=3,
                                                 topic_prefix=False,
                                                 word_length=10,
                                                 separator=", ")
    st.write(topic_model.set_topic_labels(topic_labels))
 
    st.write(good_model.visualize_topics())

    st.write(good_model.visualize_barchart())

    st.write(good_model.visualize_heatmap())
    
    doc_num = int(st.number_input('enter the number of topic to explore', value= 0))
    
    st.write(good_model.get_representative_docs(doc_num))
    # pros
    good_topic_info = good_model.get_topic_info()
    
    if len(good_reviews) < 300:
      good_topic_info['percentage'] = good_topic_info['Count'].apply(lambda x: (x / good_topic_info['Count'].sum()) * 100/4)
    else:
      good_topic_info['percentage'] = good_topic_info['Count'].apply(lambda x: (x / good_topic_info['Count'].sum()) * 100)
    
    st.write(good_topic_info)
    good_topic_info =good_topic_info.to_csv(index=False).encode('utf-8')
    st.download_button(
     label="Download Positive Analysis",
     data=good_topic_info,
     mime='text/csv',
     file_name='Positive_analysis.csv')
else:

    """# Bad reviews model insight"""
    # Feature Engineering
    st.subheader('Negative Reviews')
    #Accounting for small dataset
    
    if len(bad_reviews) < 300: # Workaround if not enough documents https://github.com/MaartenGr/BERTopic/issues/97 , https://github.com/MaartenGr/Concept/issues/5
       bad_reviews_data.extend(3*bad_reviews_data)

        
    st.dataframe(bad_reviews)
    bad_model = topic_model.fit(bad_reviews_data)
    # Topics
    
    st.write(bad_model.get_topic_info())
    doc_num_2 = int(st.number_input('enter the number of topic to explore', key= 2, value= 0))
    # Labels
    st.write(bad_model.generate_topic_labels(nr_words=6, separator=", "))
    # Representative docs
    st.write(bad_model.get_representative_docs(doc_num_2))
    
    # Topic visualization
    # Bar chart
    st.write(bad_model.visualize_barchart())
    # Term Rank
    st.write(bad_model.visualize_term_rank())
    # Heatmap
    bad_model.visualize_heatmap()
    # cons
    bad_model.generate_topic_labels(nr_words=6, separator=',')
    bad_model.get_representative_docs(doc_num)

    bad_topic_info = bad_model.get_topic_info()
    if len(bad_reviews) < 300:
       bad_topic_info['percentage'] = bad_topic_info['Count'].apply(lambda x: (x / bad_topic_info['Count'].sum()) * 100/4)
    else:
       bad_topic_info['percentage'] = bad_topic_info['Count'].apply(lambda x: (x / bad_topic_info['Count'].sum()) * 100)
    
    st.write(bad_topic_info)
    bad_topic_info =bad_topic_info.to_csv(index=False).encode('utf-8')
    st.download_button(
     label="Download Negative Analysis",
     data=bad_topic_info,
     mime='text/csv',
     file_name='negative_analysis.csv')

final_df.drop(['TextBlob_Polarity'], axis= 1, inplace = True)

st.write(final_df)

final_df =final_df.to_csv(index=False).encode('utf-8')


st.download_button(
     label="Download Dataframe analysis",
     data=final_df,
     mime='text/csv',
     file_name='full_data_analysis.csv')

brief_text="there is a total of {num_reviews} for the product {asin_num} of those, there are {num_en} english reviews .there are {positive_num} positive reviews and {negative_num} negative reviews.".format(
    num_reviews= len(df),
    asin_num = df['asin'].unique(),
    num_en= len(df[df['detect']=='en']),
    positive_num= len(good_reviews),
    negative_num= len(bad_reviews))
st.write(brief_text)
