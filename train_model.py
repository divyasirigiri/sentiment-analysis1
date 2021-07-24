import numpy as np
import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

nlp = spacy.load('en_core_web_sm')

# Reading the files
pos_rev = pd.read_csv('pos.txt' , sep = '\n', encoding = 'latin-1' , header = None)
pos_rev['mood'] = 1
# renaming a column
pos_rev.rename(columns = {0:'review'} , inplace = True)
pos_rev["new_review"]=""
k=0
for i in pos_rev["review"]:
    list1=[]
    doc = nlp(i)
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not token.is_digit:
                    list1.append(str(token.lemma_))
                
    pos_rev["new_review"][k] = " ".join(list1)
    k+=1
    
pos_rev['new_review'] = pos_rev['new_review'].str.replace('\d+', '')
del pos_rev["review"]

neg_rev = pd.read_csv('negative.txt' , sep = '\n', encoding = 'latin-1' , header = None)
neg_rev['mood'] = 0
# renaming a column
neg_rev.rename(columns = {0:'review'} , inplace = True)
neg_rev["new_review"]=""
k=0
for i in neg_rev["review"]:
    list1=[]
    doc = nlp(i)
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not token.is_digit:
                    list1.append(str(token.lemma_))
                
    neg_rev["new_review"][k] = " ".join(list1)
    k+=1
    
del neg_rev["review"]
neg_rev['new_review'] = neg_rev['new_review'].str.replace('\d+', '')    
           

# connecting both the dataset
com_rev = pd.concat([pos_rev , neg_rev] , axis = 0).reset_index()

# train_test_split
X_train , X_test , y_train , y_test = train_test_split(com_rev['new_review'].values , com_rev['mood'].values , test_size = 0.3 , random_state = 101)
train_data = pd.DataFrame({'new_review' : X_train , 'mood':y_train})
test_data = pd.DataFrame({'new_review' : X_test , 'mood':y_test})

vector = TfidfVectorizer()
train_vectors = vector.fit_transform(train_data['new_review'])
test_vector = vector.transform(test_data['new_review'])

from sklearn.svm import SVC
classifier  = SVC()
classifier.fit(train_vectors , train_data['mood'])

print('Model Training is done')

joblib.dump(classifier , 'sentiment_analysis_model.pkl')
joblib.dump(vector , 'vector.pkl')

