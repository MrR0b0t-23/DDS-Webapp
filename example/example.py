# -*- coding: utf-8 -*-

import nltk
import pickle
import numpy as np
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfTransformer

with open("CPS.pkl", 'rb') as file:  
    CPS_Model = pickle.load(file)
    
class RakeImpl:
    def __init__(self, text):
        self.text = text
        self.rake = Rake()
    def getKeywords(self):
        self.rake.extract_keywords_from_text(self.text)
        return self.rake.get_ranked_phrases()
    
def random_resp(df):
    text = df['0'].sample(n=1)
    text = np.array(text)
    return (''.join(text))


 
def preprocess(user_text):
    porter = nltk.PorterStemmer()
    user_text = porter.stem(user_text)
    WNlemma = nltk.WordNetLemmatizer()
    user_text= WNlemma.lemmatize(user_text)
    user_text = ''.join(user_text)
    rakeImpl = RakeImpl(user_text)
    user_text = rakeImpl.getKeywords()[:20]
    user_text = ''.join(user_text)
    print(user_text)
    return user_text
        

def predict (symp):
    symp = preprocess(symp)
    y_pred = CPS_Model.predict_proba([symp])
    pred = CPS_Model.predict([symp])
    return pred
            

###   importing machine learning model  ###

    

symp = "cough runny nose bodypain headache"
val = predict(symp)
print(val)