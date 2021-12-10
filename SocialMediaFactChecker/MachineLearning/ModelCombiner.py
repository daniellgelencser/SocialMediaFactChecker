################################################################################################################################
# File combined 2 models 
# model1: decision tree model , not included in final results, decisionTreeModel.pkl
# model1 was not included in final results since we did nto feed in full tweents was not effective with judging punctionation
# model2: navie_bayes classifer, twitterTFModel.pkl this name is misleading as it was trained on news article data
# model combiner original displaed false if models did not agree. 
# To avoid type2 error want to return False if models deagreed
#################################################################################################################################

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')

from bs4 import BeautifulSoup
import re,string
import pickle

def prepare_test(text):
    data_small = pd.DataFrame(np.array([text]),columns=['text'])

    special_characters = '!?@#$%^&*()-+_=,<>/'

    data_small['text_character_cnt'] = data_small['text'].str.len()
    data_small['text_word_cnt'] = data_small['text'].str.split().str.len()

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
        self.model = pickle.load(open("MachineLearning/decisionTreeModel.pkl",'rb'))
    
    def preditWithText(self, text):
        return int(self.model.predict(prepare_test(text).drop(columns=['text']))[0])
    
    def return_str(self, text):
        if self.preditWithText(text) == 0:
            return False
        else:
            return True

class model1_tweets:
    def __init__(self):
        self.model = pickle.load(open("MachineLearning/twitterTFModel.pkl",'rb'))
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