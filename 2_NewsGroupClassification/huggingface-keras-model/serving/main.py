from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
import os
import re
import io
import logging
import time
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchsummary import summary
import transformers as ppb #!python -m pip install transformers
from konduit.load import client_from_file

logging.basicConfig(level='DEBUG')
logging.info("Test")


class Serving(Screen):
    
    serverConfigured = False
    MAX_LEN = 512
    stop_words = ["from", "to", "subject", "title", "request", "looking", "look", "forward", "cheers", "regards", "thank", "thanks", "hi", "all", "since", "mentioned", "free", "ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"]

    def remove_stop_words(self, input_str):

        tokenized_words = input_str.split()

        filtered_words = [w for w in tokenized_words if not w in self.stop_words]

        output = " ".join(filtered_words)

        if len(output) > self.MAX_LEN:
            return output[0: self.MAX_LEN]

        return output  #return as string
    
    def preprocess_regex(self, text):
    
        # Applies preprocessing on text

        #remove leading & end white spaces and convert text to lowercase
        text = text.strip().lower()

        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # remove punctuation marks 
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for i in text:
            if i in punctuations: 
                    text = text.replace(i, "")

        # remove the characters [\], ['] and ["]
        text = re.sub(r"\\", "", text)    
        text = re.sub(r"\'", "", text)    
        text = re.sub(r"\"", "", text)

        #remove number
        text = re.sub(r"\d+", "", text)

        return text

    def config(self):
        # For DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        ## uncomment below for  BERT instead of distilBERT
        #model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights)

        print('Load index label')
        label_path = "labelclass.pickle"
        self.labelhandler = open(label_path, 'rb')
        self.labelhandler = pickle.load(self.labelhandler)

        self.client = client_from_file("config.yaml")
        
        self.serverConfigured = True
        
        print('Configuration done')
        
    def get_model_result(self):
        
        if self.serverConfigured == False:
            self.config()
        
        
        handle = open("D:\\Users\\chiawei\\konduit\\Github\\newsgroup_data\\20news-bydate\\20news-bydate-test\\" + self.ids['dataPath'].text, 'r')
        processed_input = self.remove_stop_words(self.preprocess_regex(handle.read()))
        
        tokenized_test_data = self.tokenizer.encode(processed_input, add_special_tokens=True)

        
        max_len_add = self.MAX_LEN

        if len(tokenized_test_data) > self.MAX_LEN:
            max_len_add = len(tokenized_test_data)

        padded_test_data = np.array([tokenized_test_data + [0]*(max_len_add-len(tokenized_test_data))])

        attention_test_data = np.where(padded_test_data != 0, 1, 0)

        input_test_ids = torch.tensor(padded_test_data)  
        attention_test_mask = torch.tensor(attention_test_data)

        input_test_ids = torch.tensor(input_test_ids).to(torch.int64)

        with torch.no_grad():
            last_hidden_states = self.model(input_test_ids, attention_mask=attention_test_mask)


        test_feature = last_hidden_states[0][:,0,:].numpy()
        
        response = self.client.predict({"default": test_feature})

        results = response["output"]["probabilities"]
        index = int(np.argmax(response['output']['probabilities'], 1)[0])
        
        return "Label: " + self.labelhandler[index] + "   Confidence: " + str(np.max(response['output']['probabilities']))
        
    
    
    def predict(self):
        
        self.ids['host'].text = "http://localhost:65322"
        self.ids['prediction'].text = self.get_model_result()
        #self.ids['dataPath'].text = "misc.forsale\\76128"
        


class ServingApp(App):

    host = StringProperty(None)
    dataPath = StringProperty(None)

    def build(self):
        manager = ScreenManager()
        manager.add_widget(Serving(name='SkymindPro'))

        return manager


if __name__ == '__main__':
    ServingApp().run()
