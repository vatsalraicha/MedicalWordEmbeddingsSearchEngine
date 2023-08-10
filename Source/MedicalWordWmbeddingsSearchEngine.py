# Medical Word Embeddings Search Engine

# Importing Streamlit 

import streamlit as st

# Importing Other Librariers

import pandas as pd
import numpy as np

import string
from collections import Counter
import re

import gensim
from gensim.models import Word2Vec
from gensim.models import FastText

from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Reading the dataset

covid = pd.read_csv("../data/Dimension-covid.csv") # For Pre Processing
covid_bk = pd.read_csv("../data/Dimension-covid.csv") # For results

# Pre Processing Data

''' Setting up Stopwords, Regexp Tokenizer and Lemmatizer.'''
sw = stopwords.words("english")
tokenizer = RegexpTokenizer(r'[A-z]+')
lemmatizer = WordNetLemmatizer()

# Functions for Pre Processing Data

# 1. Convert all text to lower case
def text_tolower(text):
    return text.lower()

# 2. Function to remove URLs
def remove_urls(text):
    new_text = " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return new_text

# 3. Get rid of numbers
def remove_numbers(text):
    '''
    if text is None or "nan" or pd.NA or np.NaN:
        return text
    else:
    '''
    return re.sub(r"\d+", " ", text)

# 4. Remove Punctuation
def remove_punctuation(text):
    translator = str.maketrans("","", string.punctuation)
    return text.translate(translator)

# 5. Word Tokenizer

def tokenize(text):
    return word_tokenize(text)

# 6. Remove StopWords
def remove_stopwords(text):
    return [word for word in text if word not in sw]

# 7. Lemmatize Words
def lemmatize(text):
    return [lemmatizer.lemmatize(token) for token in text]

# 8. Function that calls the above 7 functions.
def preprocess_text(text):

    text = remove_numbers(text)
    text = text_tolower(text)
    
    text = remove_urls(text)
    text = remove_punctuation(text)

    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    
    text = " ".join(text)
    text = re.sub("\n", " ", text)
    return text

skipgram = Word2Vec.load("notebooks/skipgramx2.bin")
FastText_model = Word2Vec.load("notebooks/FastText.bin")

vector_size=100   #defining vector size for each word

# Function to build the Mean Vector
def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in words if word in list(word2vec_model.wv.index_to_key)] #if word is in vocab 
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0]*100)

K_skipgram=pd.read_csv('notebooks/skipgram-vec.csv')   #Loading our pretrained vectors of each abstract

skipgram_vectors=[]                 #transforming dataframe into required array like structure as we did in above step
for i in range(covid.shape[0]):
    skipgram_vectors.append(K_skipgram[str(i)].values)

K_fasttext=pd.read_csv('notebooks/FastText-vec.csv')   #Loading our pretrained vectors of each abstract

fast_vectors=[]                        #transforming dataframe into required array like structure as we did in above step
for i in range(covid.shape[0]):
    fast_vectors.append(K_fasttext[str(i)].values)

#defining function to define cosine similarity

from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):

    return dot(a, b)/(norm(a)*norm(b)) 

pd.set_option("display.max_colwidth", None)       #this function will display full text from each column


# Streamlit Function

def main():
    # Load data and models
    data = covid_bk     #our data which we have to display
      
    

    st.title("Clinical Trial Search engine")      #title of our app
    st.write('Select Model')       #text below title

    
    Vectors = st.selectbox("Model",options=['Skipgram' , 'Fasttext'])
    if Vectors=='Skipgram':
        K=skipgram_vectors
        word2vec_model=skipgram
    elif Vectors=='Fasttext':
        K=fast_vectors
        word2vec_model=FastText_model

    st.write('Type your query here')

    query = st.text_input("Search box")   #getting input from user

    def preprocessing_input(query):
            
            query=preprocess_text(query)
            query=query.replace('\n',' ')
            K=get_mean_vector(word2vec_model,query)
   
            return K   

    def top_n(query,p,covid_bk):
        
        query=preprocessing_input(query)   
                                    
        x=[]
    
        for i in range(len(p)):
            x.append(cos_sim(query,p[i]))
        
        tmp=list(x)    
        res = sorted(range(len(x)), key = lambda sub: x[sub])[-10:]
        sim=[tmp[i] for i in reversed(res)]
        print(sim)

        L=[]
        for i in reversed(res):   
            L.append(i)

        return covid_bk.iloc[L, [1,2,5,6]],sim  
    
    model = top_n
    if query:
        
        #print(f"query is {query}")
        #print(f"Model Selected is {K}")
        #print(f"Dataframe first row is {data.Abstract[0]}")
        P,sim =model(str(query),K,data)     #storing our output dataframe in P
        #Plotly function to display our dataframe in form of plotly table
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title','Abstract','Publication Date','Score']),
                                       cells=dict(values=[list(P['Trial ID'].values),list(P['Title'].values), 
                                                          list(P['Abstract'].values),list(P['Publication date'].values),
                                                          list(np.around(sim,4))],align=['center','right']))])
        #displaying our plotly table
        fig.update_layout(height=1700,width=700,margin=dict(l=0, r=10, t=20, b=20))
        
        st.plotly_chart(fig) 
        # Get individual results
    

if __name__ == "__main__":
    main()