# -*- coding: utf-8 -*-
import glob
import pickle
import os

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, TimeDistributed
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence

class LanguageModel(object):
    def __init__(self, data_dir):
        self.init_weights = None
        self.vcb_mapping = None
        self.PAD = 0
        self.data_dir = data_dir
        
    def get_data(self, high, low):
        src, tgt = [], []
        #Create the source and target dataset from the high and low languages
        for filename in glob.glob(self.data_dir+"/"+"*-train-high"):
            if filename.split("/")[-1] in high:
                print (filename.split("/")[-1])
                with open(filename,"r") as fp:
                    for line in fp.readlines():
                        _, word, _ = line.split("\t")
                        src += list(word),
                        tgt += list(word)[1:] + ["$"],
        for filename in glob.glob(self.data_dir+"/"+low+"-train-low"):
            print (filename.split("/")[-1])
            with open(filename,"r") as fp:
                for line in fp.readlines():
                    _, word, _ = line.split("\t")
                    src += list(word),
                    tgt += list(word)[1:] + ["$"],
        
        vcb = set()
        for inp in src:
            vcb.update(inp)
        for inp in tgt:
            vcb.update(inp)
        
        self.vcb_mapping = dict((c, i) for i, c in enumerate(list(vcb), 1))
        return src, tgt
    
    def encode(self, src):
        #Encode the source and target datasets        
        enc = []
        for inp in src:
            enc += [self.vcb_mapping[c] for c in inp],
        return enc
    
    def process_data(self, high, low):
        #Get data and encode it
        s, t = self.get_data(high, low)
        s = self.encode(s)
        t = self.encode(t)
        s = sequence.pad_sequences(s, maxlen=20, padding="post", truncating="post")
        t = sequence.pad_sequences(t, maxlen=20, padding="post", truncating="post")
        t = np.eye(len(self.vcb_mapping)+1)[t.astype('int')]
        return s,t

    def train(self, high, low):
        #Build and train the model and get the LSTM layer weights
        s, t = self.process_data(high, low)
        encoder_input = Input(shape=(20,))
        embedding_layer = Embedding(len(self.vcb_mapping)+1, 64, input_length=20)
        embedding = embedding_layer(encoder_input)
        encoder_layer = LSTM(64, return_sequences=True, unroll=True)
        encoder = encoder_layer(embedding)
        output = TimeDistributed(Dense(len(self.vcb_mapping)+1, activation="softmax"))(encoder)
        model = Model(inputs=[encoder_input], outputs=[output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        
        model.fit(x=[s[:-100]], y=[t[:-100]],
                  validation_data=([s[-100:]], [t[-100:]]),
                  batch_size=64, 
                  epochs=10)
        
        self.init_weights = encoder_layer.get_weights()

'''
if __name__ == "__main__":
    with open('lang_pair_dict.pickle', 'rb') as handle:
        lang_pair_dict = pickle.load(handle)
    lg_model = LanguageModel("/home/hari/Documents/NN_DL/Project/crosslingual-inflection/data/spanish--occitan")
    lg_model.train(low="occitan", high=lang_pair_dict["occitan"])
    weights = lg_model.init_weights
    print (weights[0].shape)
    
    print (len(lg_model.vcb_mapping))
    print (lg_model.vcb_mapping)
'''