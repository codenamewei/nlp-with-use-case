import pandas as pd
import numpy as np
import re
import nltk 
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI

app = FastAPI()

model_path = "sklearn_model.pickle"
vectorizer_path = "sklearn_vectorizer.pickle"

#load model 
model = pickle.load(open(model_path, 'rb'))

#load vectorizer
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

@app.get("/")
async def root():
    return {"message": "Server up!"}


@app.put("/predict")
async def predict():
    
    stripped_input = [preprocess("@VirginAmerica I didn't today... Must mean I need to take another trip!")]
    test_feature = vectorizer.transform(stripped_input).toarray()
    
    loaded_prediction = model.predict(test_feature)
    
    return {"response": loaded_prediction[0]}


def preprocess(feature):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(feature))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature) 

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    return processed_feature

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=10233)