import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Dataset', default='ucf101')
parser.add_argument('-c', '--cross', help='Fold cross validation index', default=1, type=int)
parser.add_argument('-s', '--segment', help='Number of segments', default=3, type=int)
parser.add_argument('-debug', '--debug', help='Number of classes', default=1, type=int)
args = parser.parse_args()
print args

import cv2
import os
import sys
import random
import numpy as np
import config
import pickle

# Lay tuy chon 
dataset = args.dataset
cross_index = args.cross
num_seq = args.segment
if args.debug == 1:
    debug = True
else:
    debug = False

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

train_file = r'data/{}-trainlist0{}.txt'.format(dataset,cross_index)
test_file = r'data/{}-testlist0{}.txt'.format(dataset,cross_index)
out_file_folder = r'{}database/'.format(output_path)
data_file = r'{}data-{}-{}-new.pickle'.format(out_file_folder,dataset,num_seq)

out_train_file = r'{}{}-train{}-split{}-new.pickle'.format(out_file_folder,dataset,num_seq,cross_index)
out_test_file = r'{}{}-test{}-split{}-new.pickle'.format(out_file_folder,dataset,num_seq,cross_index)

count = 0
with open(data_file,'rb') as f1:
    data = pickle.load(f1)

train_data = []
test_data = []
train_name = []
test_name = []

with open(train_file) as f:
    for line in f:
        arr_line = line.rstrip().split(' ')[0] # return folder/subfolder/name.mpg
        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0]
        full_name = folder_video + '/' + name_video
        train_name.append(full_name)

with open(test_file) as f:
    for line in f:
        arr_line = line.rstrip().split(' ')[0] # return folder/subfolder/name.mpg
        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0]
        full_name = folder_video + '/' + name_video
        test_name.append(full_name)

length = len(data)
data_name = []
for i in range(length):
    if (data[i][0] in train_name):
        train_data.append(data[i])
    elif (data[i][0] in test_name):
        test_data.append(data[i])

print 'Generate {} train samples for {} dataset'.format(len(train_data),dataset)
print 'Generate {} test samples for {} dataset'.format(len(test_data),dataset)

# Ghi du lieu dia chi ra file
with open(out_train_file,'wb') as f2:
    pickle.dump(train_data,f2)

with open(out_test_file,'wb') as f2:
    pickle.dump(test_data,f2)