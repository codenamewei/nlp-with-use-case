import re
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

class Input(BaseModel):
    input_str: str

model_path = "sklearn_model.pickle"
vectorizer_path = "sklearn_vectorizer.pickle"

#load model 
model = pickle.load(open(model_path, 'rb'))

#load vectorizer
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server up!"}

#example body
#{"input_str": "San Francisco to San Diego.The aircraft was clean and comfortable. The snacks and beverages were nice and the attendees are helpful."}
@app.put("/predict")
async def predict(input: Input):
    
    input_body = input.dict()
    
    input = input_body['input_str']
    stripped_input = [preprocess(input)]

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

# local run with uvicorn skdeploy:app --host 0.0.0.0 --port 5000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)