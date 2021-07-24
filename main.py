from logging import debug
from flask import Flask, render_template, request
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
import spacy
app = Flask(__name__)

import joblib
model = joblib.load('sentiment_analysis_model.pkl')
vector = joblib.load('vector.pkl')
nlp = spacy.load('en_core_web_sm')

@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/classify',methods = ['POST'])
def classify():
    ip = request.form.get('ip')
    doc = nlp(ip)
    list1=[]
    for token in doc:
        if not token.is_punct:
            if not token.is_stop:
                if not token.is_digit:
                    list1.append(str(token.lemma_))

    output = model.predict(vector.transform(list1))
    if 1 in output:
        res = "positive"
    else:
        res="negative"

    return render_template('base.html',prediction_text=f'\n the sentiment of the entered text is {res}')

if __name__=='__main__':
    app.run(debug=True)