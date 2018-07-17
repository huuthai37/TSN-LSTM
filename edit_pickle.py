import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Dataset', default='ucf101')
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
num_seq = args.segment
if args.debug == 1:
    debug = True
else:
    debug = False


def count_frames(path):
    cap = cv2.VideoCapture(path)
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break  
        i += 1
    cap.release()
    return i

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

out_file_folder = r'{}database/'.format(output_path)
data_file = r'{}data-{}-{}.pickle'.format(out_file_folder,dataset,num_seq)
data_file_out = r'{}data-{}-{}-new.pickle'.format(out_file_folder,dataset,num_seq)

count = 0
with open(data_file,'rb') as f1:
    data = pickle.load(f1)

length_data = len(data)

for l in range(length_data):
    
    path_video = data[l][0]
    leng = count_frames(data_input_folder + path_video + '.avi')
    if leng == 0:
        print 'error'
        sys.exit()
    data[l].append(leng)

    if l%500 == 0:
        print (l, leng)

with open(data_file_out,'wb') as f2:
    pickle.dump(data,f2)



