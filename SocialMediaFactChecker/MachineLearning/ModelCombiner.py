# %%
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np



# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.feature_extraction.text import TfidfTransformer

from keras.preprocessing import text,sequence
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.util import ngrams
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

import os
import json
import re
import string
import warnings
import numpy
import pandas
import matplotlib.pyplot as pyplot
import seaborn
import nltk
import gensim
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.pipeline import Pipeline



def prepare_test(text):
    data_small = pd.DataFrame(np.array([text]),columns=['text'])

    special_characters = '!?@#$%^&*()-+_=,<>/'

    data_small['text_character_cnt'] = data_small['text'].str.len()
    # if(data['text_character_cnt'] > 0):
    data_small['text_word_cnt'] = data_small['text'].str.split().str.len()
    # data['text_character_per_word'] = data['text_character_cnt'] / data['text_word_cnt']

    data_small['text_special_cnt'] = data_small['text'].apply(lambda x: len([x for x in x.split() if any(char in special_characters for char in x)]))

    for char in special_characters:
        data_small['text_' + char + '_per_char'] = data_small['text'].apply(lambda x: len([x for x in x.split() if char in x]))
        data_small['text_' + char + '_per_word'] = data_small['text'].apply(lambda x: len([x for x in x.split() if char in x]))

    data_small['text_http_cnt'] = data_small['text'].apply(lambda x: len([x for x in x.split() if 'http' in x]))
    data_small['text_www_cnt'] = data_small['text'].apply(lambda x: len([x for x in x.split() if 'www' in x]))
    data_small['text_number_cnt'] = data_small['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    
    return data_small

def string_html(text):
    soup=BeautifulSoup(text,"html.parser")
    return soup.get_text()

def remove_square_brackets(text):
    return re.sub('\[[^]]*\]','',text)

def remove_URL(text):
    return re.sub(r'http\S+','',text)

# def remove_stopwords(text):
#     final_text=[]
#     for i in text.split():
#         if i.strip().lower() not in stop_words:
#             final_text.append(i.strip())
#     return " ".join(final_text)

def remove_HashOrAT(text):
    return re.sub('@|#','',text)

def remove_puncuation(text):
    return text.translate(str.maketrans('','',string.punctuation))

def remove_uppercase(text):
    return text.lower()
    

def clean_text_data(text):
    text=string_html(text)
    text=remove_square_brackets(text)
    text=remove_URL(text)
    text=remove_HashOrAT(text)
    text=remove_puncuation(text)
    text=remove_uppercase(text)
    return text

class model1_News:
    def __init__(self):
        self.model = pickle.load(open("decisionTreeModel.pkl",'rb'))
    
    def preditWithText(self, text):
        return int(self.model.predict(prepare_test(text).drop(columns=['text']))[0])
    
    def return_str(self, text):
        if self.preditWithText(text) == 0:
            return False
        else:
            return True

class model1_tweets:
    def __init__(self):
        self.model = pickle.load(open("twitterTFModel.pkl",'rb'))
        self.key = realorfake = {"True":1,"False":0}
        
    
    def preditWithText(self, text):
        return self.model.predict([text])[0]

    def return_str(self, text):
        if self.preditWithText(text) == 0:
            return False
        else:
            return True

model1 = model1_tweets()
model2 = model1_News()
        

class joinmodel:
    def __init__(self):
        self.model1 = model1_tweets()
        self.model2 = model1_News()
        self.lastResult = None
    def getTrueFalse(self, text, verbose = False):
        result = {}
        if type(text) != str:
            return "Error please feed in a string"
        
        result["modelTweets_1"] = self.getTweetModel_1(text)
        result["modelNews_2"] = self.getNewsModel_2(text)
        result["combinedResult"] = self.getCombinedResults(result.get("modelTweets_1"), result.get("modelNews_2")) 
        self.lastResult = result
        
        if verbose:
            return result
        else:
            return result.get("combinedResult")
        
    
    def getTweetModel_1(self, text):
        return self.model1.return_str(text)
    def getNewsModel_2(self, text):
        return self.model2.return_str(text)
    
    # this will be replaced with a lasso regression model trained with both models but this is a simple implemtation
    # if both results are the same then it returns that boolean. otherwise it returns False
    def getCombinedResults(self, model1_result, model2_result):
        if model1_result == model2_result:
            return model2_result
        else: 
            return False





