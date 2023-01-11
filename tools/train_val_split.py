import cv2
import os
import shutil
import numpy as np
import pickle
from tqdm import trange
data_root = '/data/preprocessed_FAIR1m15_512_ms'
overlap = 512
size = 1024
ms = '0.5-1.0-1.5'
fold = 9 #0-10
train = 'train'
train_dir = f'{data_root}/{train}_{size}_{overlap}_{ms}'
val_dir = f'{data_root}/val_{size}_{overlap}_{ms}'
os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_dir + '/images', exist_ok=True)
os.makedirs(val_dir + '/labelTxt', exist_ok=True)

imgdir = sorted(os.listdir(train_dir + '/images'))
xmldir = sorted(os.listdir(train_dir + '/labelTxt'))
for x in trange(len(imgdir)):
    if x % 10 < fold:
        continue
    else:
        shutil.copy(train_dir + '/images/' + imgdir[x],val_dir + '/images')
        shutil.copy(train_dir + '/labelTxt/' + xmldir[x],val_dir + '/labelTxt')
pkl_file = open(train_dir + '/labels.pkl', 'rb')
pkl = pickle.load(pkl_file)
val_img_dir = sorted(os.listdir(val_dir + '/images'))
val_pkl = []
val_pkl_file = val_dir + '/labels.pkl'
for i in trange(len(pkl)):
    if pkl[i]['filename'] not in val_img_dir:
        continue
    else:
        val_pkl.append(pkl[i])

pickle.dump(val_pkl, open(val_pkl_file, "wb"))


# pkl_file1 = open('/data/preprocessed_DOTA2/train_1024_200_1.0/labels.pkl', 'rb')
# pkl1 = pickle.load(pkl_file1)
# pkl_file2 = open('/data/preprocessed_DOTA2/val_1024_200_1.0/labels.pkl', 'rb')
# pkl2 = pickle.load(pkl_file2)
# pkl_file3 = open('/work_dirs/s2anet_swint_fpn1x_dota2/detections/val_1/val.pkl', 'rb')
# pkl3 = pickle.load(pkl_file3)
# trnimgdir = sorted(os.listdir('/data/preprocessed_DOTA2/train_1024_200_1.0/images'))
# trnxmldir = sorted(os.listdir('/data/preprocessed_DOTA2/train_1024_200_1.0/labelTxt'))
# valimgdir = sorted(os.listdir('/data/preprocessed_DOTA2/val_1024_200_1.0/images'))
# valxmldir = sorted(os.listdir('/data/preprocessed_DOTA2/val_1024_200_1.0/labelTxt'))
#
# fold = 9 #0-10
# train_dir = '/data/preprocessed_DOTA1/train_1024_256_1.0'
# val_dir = '/data/preprocessed_DOTA1/val_1024_256_1.0'
# a = os.listdir(train_dir + '/images')
# b = os.listdir(train_dir + '/labelTxt')
# c = os.listdir(val_dir + '/images')
# d = os.listdir(val_dir + '/labelTxt')
# pkl_file1 = open('/data/preprocessed_DOTA1/train_1024_256_1.0/labels.pkl', 'rb')
# pkl1 = pickle.load(pkl_file1)
# pkl_file2 = open('/data/preprocessed_DOTA1/val_1024_256_1.0/labels.pkl', 'rb')
# pkl2 = pickle.load(pkl_file2)
# for i in trange(len(pkl1)):
#     if pkl1[i]['ann']
# e=1
# os.makedirs(val_dir, exist_ok=True)
# os.makedirs(val_dir + '/images', exist_ok=True)
# os.makedirs(val_dir + '/labelTxt', exist_ok=True)
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(train_dir + '/images', exist_ok=True)
# os.makedirs(train_dir + '/labelTxt', exist_ok=True)
# ori_dir = '/data/data/processed_DOTA1/trainval_1024_256_1.0'
# pre_dir = '/data/preprocessed_DOTA1'
# ori_test_dir = '/data/data/processed_DOTA1/test_1024_200_1.0'
# test_dir = '/data/preprocessed_DOTA1/test_1024_200_1.0'
# shutil.copytree(ori_test_dir,test_dir)
# imgdir = sorted(os.listdir(ori_dir + '/images'))
# xmldir = sorted(os.listdir(ori_dir + '/labelTxt'))
# for x in trange(len(imgdir)):
#     if x % 10 < fold:
#         shutil.copy(ori_dir + '/images/' + imgdir[x], train_dir + '/images')
#         shutil.copy(ori_dir + '/labelTxt/' + xmldir[x], train_dir + '/labelTxt')
#     else:
#         shutil.copy(ori_dir + '/images/' + imgdir[x],val_dir + '/images')
#         shutil.copy(ori_dir + '/labelTxt/' + xmldir[x],val_dir + '/labelTxt')
# pkl_file = open(ori_dir + '/labels.pkl', 'rb')
# pkl = pickle.load(pkl_file)
# train_img_dir = sorted(os.listdir(pre_dir + '/train_1024_256_1.0/images'))
# val_img_dir = sorted(os.listdir(pre_dir + '/val_1024_256_1.0/images'))
# train_pkl,val_pkl = [],[]
# train_pkl_file = pre_dir + '/train_1024_256_1.0/labels.pkl'
# val_pkl_file = pre_dir + '/val_1024_256_1.0/labels.pkl'
# for i in trange(len(pkl)):
#     if pkl[i]['filename'] not in val_img_dir:
#         train_pkl.append(pkl[i])
#     else:
#         val_pkl.append(pkl[i])
#



a =1