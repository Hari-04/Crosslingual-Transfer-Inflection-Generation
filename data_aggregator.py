# -*- coding: utf-8 -*-

from shutil import copyfile
import glob
import os
from collections import defaultdict
import pickle

src = "/home/hari/Documents/NN_DL/Project/crosslingual-inflection/data"
dst = "/home/hari/Documents/NN_DL/Project/crosslingual-inflection/total_data"

lang_pair_dict = defaultdict(list)
lang_pairs = [x[0] for x in os.walk(src)]
for pair in lang_pairs:
    lg = pair.split("/")[-1].split("--")
    if len(lg) > 1:
        lang_pair_dict[lg[1]] += lg[0]+"-train-high",

cp_files = set()
for dir_name in lang_pairs:
    print (dir_name)
    for filename in glob.glob(dir_name+"/*-train-*"):
        print ("filename:",filename)
        if filename.split("/")[-1] not in cp_files:
            copyfile(filename, dst+"/"+filename.split("/")[-1])
            cp_files.add(filename.split("/")[-1])
    for filename in glob.glob(dir_name+"/*-dev*"):
        if filename.split("/")[-1] not in cp_files:
            copyfile(filename, dst+"/"+filename.split("/")[-1])
            cp_files.add(filename.split("/")[-1])

print (cp_files)
with open('lang_pair_dict.pickle', 'wb') as handle:
    pickle.dump(lang_pair_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)