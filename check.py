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

# Cau hinh folder du lieu
server = config.server()
data_input_folder = config.data_input_path()
output_path = config.data_output_path()

out_file_folder = r'{}database/'.format(output_path)
data_file = r'{}data-{}-{}.pickle'.format(out_file_folder,dataset,num_seq)

count = 0
with open(data_file,'rb') as f1:
    data = pickle.load(f1)

length_data = len(data)
data_folder_opt = '/mnt/ucf101/tvl1_flow/'
data_folder_seq_opt = r'{}{}-seq-opt/'.format(output_path,dataset)

for l in range(length_data):
    if l <= 9199:
        continue
    path_video = data[l][0]
    render_opt = data[l][1]
    name_video = path_video.split('/')[1]
    u = data_folder_opt + 'u/' + path_video + '/frame'
    v = data_folder_opt + 'v/' + path_video + '/frame'

    return_data = []

    if (render_opt[0] >= 0):
        render = render_opt
    else:
        render = [render_opt[1]]
    len_render_opt = len(render)

    for k in range(len_render_opt):
        for i in range(10):
            if (not os.path.exists(u + str(render[k]/2 + 1 + i).zfill(6) + '.jpg')):
                print(u + str(render[k]/2 + 1 + i).zfill(6) + '.jpg')
            if (not os.path.exists(v + str(render[k]/2 + 1 + i).zfill(6) + '.jpg')):
                print(v + str(render[k]/2 + 1 + i).zfill(6) + '.jpg')

    if l%500 == 0:
        print l



