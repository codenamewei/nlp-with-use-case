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

from konduit.load import client_from_file

logging.basicConfig(level='DEBUG')
logging.info("Test")


class Serving(Screen):
    
    serverConfigured = False

    def config(self):
        tokenizer_path = 'tokenizer.pickle'
        handle = open(tokenizer_path, 'rb')
        self.tokenizer = pickle.load(handle)

        print('Load index label')
        label_path = "labelclass.pickle"
        self.labelhandler = open(label_path, 'rb')
        self.labelhandler = pickle.load(self.labelhandler)

        self.MAX_LEN = 256
        self.client = client_from_file("config.yaml")
        
        self.serverConfigured = True
        
        print('Configuration done')
        
    def get_model_result(self):
        
        if self.serverConfigured == False:
            self.config()
        
        
        handle = open("D:\\Users\\chiawei\\konduit\\Github\\newsgroup_data\\20news-bydate\\20news-bydate-test\\" + self.ids['dataPath'].text, 'r')
        input_data = list([handle.read()])
        input_sequence = self.tokenizer.texts_to_sequences(input_data)
        input = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen = self.MAX_LEN, padding = "post")
        
        response = self.client.predict({"default": input})

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
