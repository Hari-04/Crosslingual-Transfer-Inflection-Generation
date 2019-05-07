# -*- coding: utf-8 -*-
import os
import sys

import importlib as imp
import pickle
import numpy as np

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from keras.layers import Activation, dot, concatenate
from keras.models import Model, load_model
from keras import backend as K

import datagenerator, language_model

imp.reload(datagenerator)
imp.reload(language_model)

class CLTMI(object):
    def __init__(self, mode, low, high=None):
        with open('lang_pair_dict.pickle', 'rb') as handle:
                lang_pair_dict = pickle.load(handle)
        self.low = low; self.high = lang_pair_dict[low]
        self.tr_src, self.tr_tgt, self.te_src, self.te_tgt = None, None, None, None
        
        if mode == "all":
            #Train with low and all corresponding high language data
            self.data_dir = os.getcwd()+"/total_data"
        else:            
            #Train with the given high-low dataset
            self.data_dir = os.getcwd()+"/data/"+high+"--"+low
        
        #Create data and lg_model objects for furthur use
        self.data = datagenerator.DataGenerator(self.data_dir)
        self.lg_model = language_model.LanguageModel(data_dir=self.data_dir)
        
        #Train the language model
        print ("Training Language Model for Weights.....")
        self.lg_model.train(high=self.high, low=self.low)
        
    def rec_init_weights(self, shape, dtype=float):
        #Send Recurrent weights from the trained language model
        #print ("Weight initialization..!!!",shape)
        return self.lg_model.init_weights[0] #K.constant(0, shape=shape, dtype=dtype)
    
    def ker_init_weights(self, shape, dtype=float):
        #Send kernal weights from the trained language model
        #print ("Weight initialization..!!!",shape)
        return self.lg_model.init_weights[1] #K.random_normal(shape, dtype=dtype)
    
    def get_data(self):
        #Get training data
        tr_src, tr_tgt = self.data.build_data("train", low=self.low, high=self.high)
        tr_src = self.data.transform_data("src",tr_src)
        tr_tgt = self.data.transform_data("tgt",tr_tgt)
        
        te_src, te_tgt = self.data.build_data("dev", low=self.low)
        te_src = self.data.transform_data("src",te_src)
        te_tgt = self.data.transform_data("tgt",te_tgt)
        
        self.input_dict_size = len(self.data.src_encoding)+3
        self.output_dict_size = len(self.data.tgt_encoding)+3
        self.in_length, self.out_length = 20, 20
        self.tr_src, self.tr_tgt, self.te_src, self.te_tgt = tr_src, tr_tgt, te_src, te_tgt

    def build_model(self):
        #Build the RNN Encoder-Decoder Model with Attention
        encoder_input = Input(shape=(self.in_length,))
        encoder_embedding = Embedding(self.input_dict_size, 64, input_length=self.in_length, mask_zero=True)(encoder_input)
        encoder_lstm_0 = LSTM(64, return_sequences=True, unroll=True)(encoder_embedding)
        encoder_lstm = LSTM(64, return_sequences=True, unroll=True, recurrent_initializer=self.rec_init_weights, kernel_initializer=self.ker_init_weights)(encoder_lstm_0) #recurrent_initializer=rec_init_weights, kernel_initializer=ker_init_weights
        encoder_vector = encoder_lstm[:,-1,:]        
        decoder_input = Input(shape=(self.out_length,))
        decoder_embedding = Embedding(self.output_dict_size, 64, input_length=self.out_length, mask_zero=True)(decoder_input)
        decoder_lstm_0 = LSTM(64, return_sequences=True, unroll=True)(decoder_embedding)
        decoder_lstm = LSTM(64, return_sequences=True, unroll=True, recurrent_initializer=self.rec_init_weights, kernel_initializer=self.ker_init_weights) #, recurrent_initializer=init_weights, kernel_initializer=init_weights
        decoder_lstm = decoder_lstm(decoder_lstm_0, initial_state=[encoder_vector, encoder_vector])        
        
        attn_matrix = dot([decoder_lstm, encoder_lstm], axes=[2, 2])
        attn = Activation('softmax')(attn_matrix)
        context_matrix = dot([attn, encoder_lstm], axes=[2,1])
        context_decoder = concatenate([context_matrix, decoder_lstm])
        out = TimeDistributed(Dense(64, activation="tanh"))(context_decoder)
        out = TimeDistributed(Dense(self.output_dict_size, activation="softmax"))(out)
        
        self.model = Model(inputs=[encoder_input, decoder_input], outputs=[out])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#from keras.utils import plot_model
#plot_model(model, to_file='model_graph.png')

    def train_model(self):
        #Input output for both encoder and decoder
        tr_enc_src = self.tr_src; te_enc_src = self.te_src
        tr_dec_src = np.concatenate((np.ones((self.tr_tgt.shape[0],1)), self.tr_tgt[:,:-1]), axis=1)
        te_dec_src = np.concatenate((np.ones((self.te_tgt.shape[0],1)), self.te_tgt[:,:-1]), axis=1)
        tr_dec_tgt = np.eye(self.output_dict_size)[self.tr_tgt.astype('int')]
        te_dec_tgt = np.eye(self.output_dict_size)[self.te_tgt.astype('int')]
        
        #Train the model
        self.model.fit(x=[tr_enc_src, tr_dec_src], 
                       y=[tr_dec_tgt], 
                       validation_data=([te_enc_src, te_dec_src], [te_dec_tgt]), 
                       batch_size=64,
                       epochs=30)
    
    def predict(self, text):
        #Fit the output to the model and predict the output
        enc_src = self.data.transform_data("src",[text])
        dec_src = np.concatenate((np.ones((len(enc_src),1)), np.zeros((len(enc_src), self.in_length-1))), axis=1)
        for i in range(1, self.out_length):
            output = self.model.predict([enc_src, dec_src]).argmax(axis=2)
            dec_src[:,i] = output[:,i]
        text = ''
        for i in dec_src[0,1:]:
            if i == 0:
                break
            text += self.data.tgt_decoding[i]
        
        return text

if __name__ == "__main__":
    print (sys.argv)
    mode, low = sys.argv[1:3]
    high = None
    if len(sys.argv) > 3:
        high = sys.argv[3]
    
    #Formulate the dataset; Create and train the model
    model = CLTMI(mode, low, high)
    model.get_data()
    model.build_model()
    model.train_model()
    
    #Generate the output for the corresponding low language
    with open(os.getcwd()+"/total_data/"+low+"-dev","r") as f:
        out = []
        for line in f:
            word, lemma, attr = line.strip().split("\t")
            out += model.predict(list(word)+attr.split(";")),
            print ("test word:",word, lemma, out[-1])
    
    f = open("output_"+low+".txt", "w")
    for word in out:
        f.write(word)
        f.write("\n")
    f.flush()
    f.close()