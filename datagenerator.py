# -*- coding: utf-8 -*-
import glob
import numpy as np
import pickle

class DataGenerator(object):
    def __init__(self, path):
        self.data_dir = path
        self.PAD = 0; self.START = 1; self.UNK = 2
        self.src_vcb, self.tgt_vcb = set(), set()
        self.src_train, self.tgt_train = [], [] 
        self.src_test, self.tgt_test = [], [] 
        self.src_encoding, self.src_decoding = {}, {1:"^"}
        self.tgt_encoding, self.tgt_decoding = {}, {1:"^"}
        
    def build_data(self, dataset_type, low, high=None):
        #Create the source and target data along with their corresponding encoding
        src_data, tgt_data = [], []
        if dataset_type == "train":
            for filename in glob.glob(self.data_dir+"/*-train-*"):
                if filename.split("/")[-1] in high or filename.split("/")[-1] == low+"-train-low":                
                    print (filename)
                    with open(filename, "r") as input_file:
                        for line in input_file:
                            line = line.strip().split("\t")
                            line_src = list(line[0]) + line[2].split(";")
                            line_tgt = list(line[1])
                            src_data += line_src,
                            tgt_data += line_tgt,
                            self.src_vcb.update(line_src)
                            self.tgt_vcb.update(line_tgt)
                            
            count = 3
            for c in self.src_vcb:
                self.src_encoding[c] = count
                self.src_decoding[count] = c
                count += 1
        
            count = 3
            for c in self.tgt_vcb:
                self.tgt_encoding[c] = count
                self.tgt_decoding[count] = c
                count += 1
        
        if dataset_type == "dev":
            for filename in glob.glob(self.data_dir+"/"+low+"-dev"):
                print (filename)
                with open(filename, "r") as input_file:
                    for line in input_file:
                        line = line.strip().split("\t")
                        src_data += list(line[0]) + line[2].split(";"),
                        tgt_data += list(line[1]),
        
        return src_data, tgt_data

    def transform_data(self, dataset_type, data, length=20):
        #Encode the data
        if dataset_type == "src": 
            encoding = self.src_encoding
        else:
            encoding = self.tgt_encoding
            
        transformed_data = np.zeros(shape=(len(data), length))
        for i in range(len(data)):
            #if dataset_type!="src": transformed_data[i][0] = 1
            for j in range(min(len(data[i]), length)):
                transformed_data[i][j] = encoding.get(data[i][j], self.UNK)
        return transformed_data

'''
if __name__ == "__main__":
    with open('lang_pair_dict.pickle', 'rb') as handle:
        lang_pair_dict = pickle.load(handle)
    print (lang_pair_dict["murrinhpatha"])
    data = DataGenerator("/home/hari/Documents/NN_DL/Project/keras-encoder-decoder-attention/data")
    src, tgt = data.build_data("train", low="murrinhpatha", high=lang_pair_dict["murrinhpatha"])
    print (src[0])
    tr_data = data.transform_data("src",src[0:10])
    print (tr_data[0], len(src), len(tgt))
    src, tgt = data.build_data("dev", low="murrinhpatha", high=lang_pair_dict["murrinhpatha"])
    print (src[0])
    tr_data = data.transform_data("src",src[0:10])
    print (tr_data[0], len(src), len(tgt))
    print (len(data.src_encoding), data.src_encoding)
    print (len(data.tgt_encoding), data.tgt_encoding)
'''