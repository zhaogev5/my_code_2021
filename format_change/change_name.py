import os
from tqdm import tqdm
jpg_path = './SSDD/JPEGImages/'
anno_path = './SSDD/Annotations/'

jpg_list = os.listdir(jpg_path) 
anno_list = os.listdir(anno_path)

for i,name in enumerate(zip(jpg_list,anno_list)):
    jpg_used_name = jpg_path + name[0]
    anno_used_name = anno_path + name[1]
    # jpg_new_name = jpg_path + '4' + str(i+1).zfill(5) +'.jpg'
    anno_new_name = anno_path + '4' + str(i+1).zfill(5) +'.xml'

    os.rename(anno_used_name,anno_new_name)
    # os.rename(jpg_used_name,jpg_new_name)