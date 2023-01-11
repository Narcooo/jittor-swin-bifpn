import os
import shutil
from tqdm import trange
import pickle
# train_dir = '/data/preprocessed_DOTA1/train_1024_256_1.0'
# val_dir = '/data/preprocessed_DOTA1/val_1024_256_1.0'
# imgdir = os.listdir(train_dir + '/images')
# xmldir = os.listdir(train_dir + '/labelTxt')
# os.makedirs('/data/DOTA1dataset/train/labelTxt', exist_ok=True)
# os.makedirs('/data/DOTA1dataset/val/labelTxt', exist_ok=True)
# os.makedirs('/data/DOTA1dataset/train/labelTxt', exist_ok=True)
# os.makedirs('/data/DOTA1dataset/val/labelTxt', exist_ok=True)
# for i in trange(len(xmldir)):
#     if os.path.getsize(train_dir + '/labelTxt/' + xmldir[i]) > 0:
#         shutil.copy(train_dir + '/labelTxt/' + xmldir[i],'/data/DOTA1dataset/train/labelTxt')
# nx_dir = os.listdir('/data/DOTA1dataset/train/labelTxt')
# def filter_imgs(img_infos, min_size=-1):
#     return [img_info for img_info in img_infos
#             if (len(img_info["ann"]["bboxes"]) > 0 and min(img_info['width'], img_info['height']) >= min_size)]
#

# pkl_file1 = open('/data/DOTA1/train_1024_256_1.0/labels.pkl', 'rb')
# pkl_file1 = open('/work_dirs/faster_rcnn_RoITrans_swint_fpn_1x_dota_1_plus_with_flip_rotate_balance/checkpoints/ckpt_7.pkl', 'rb')
# pkl1 = pickle.load(pkl_file1)
# pkl1 = filter_imgs(pkl1)
# b = pkl1[13470]
# c = pkl1[13469]
# #13470,
# pkl3 = pkl1[:13460]
# pkl2 = pkl1[13480:]
# pkl3.extend(pkl2)
# # pkl3 = pkl3[13400:]
# pickle.dump(pkl3, open('/data/preprocessed_DOTA1/train_1024_256_1.0/labels.pkl', "wb"))

pkl_file1 = open('/work_dirs/faster_rcnn_RoITrans_swint_fpn_1x_dota_1_plus_with_flip_rotate_balance/checkpoints/ckpt_7.pkl', 'rb')
pkl1 = pickle.load(pkl_file1)
a=1