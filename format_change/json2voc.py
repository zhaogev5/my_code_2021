import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

defect_name2label = {
    'ship': 1, 
}

# 1.标签路径
#image_path1 = "F:/guangdong/guangdong1_round1_train1_20190818/defect_Images/"  # 原始labelme标注数据路径
#image_path2 = "F:/guangdong/guangdong1_round1_train2_20190828/defect_Images/"  # 原始labelme标注数据路径
saved_path = "results/VOC2012/"  # 保存路径

# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

json_file = "my_results.bbox.json" #比赛json格式路径
# json_file2 = "F:/guangdong/guangdong1_round1_train2_20190828/Annotations/anno_train.json"
files = [1]
# 4.读取标注信息并写入 xml
# for json_file in [json_file1, json_file2]:
anno_result = pd.read_json(open(json_file, "r"))
file_name_list = list(set(anno_result['image_id']))
print(file_name_list)
for file_name in tqdm(file_name_list):

    with codecs.open(saved_path + "Annotations/" + str(file_name) + ".xml", "w", "utf-8") as xml:
        #height, width, channels = 1000, 2446, 3
        # xml.write('<?xml version="1.0" encoding="utf-8"?>\n')
        xml.write('<annotation>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<filename>'+str(file_name)+'.tif</filename>\n')
        xml.write('\t\t<origin>GF2/GF3</origin>\n')
        xml.write('\t</source>\n')
        xml.write('\t<research>\n')
        xml.write('\t\t<version>4.0</version>\n')
        xml.write('\t\t<provider>xd</provider>\n')
        xml.write('\t\t<author>ymhj</author>\n')
        xml.write('\t\t<pluginname>Detection</pluginname>\n')
        xml.write('\t\t<time>2020-07-2020-11</time>\n')
        xml.write('\t<research>\n')

        answer = anno_result[anno_result['image_id'] == file_name]
        xml.write('\t<objects>\n')
        for box, score in zip(answer['bbox'], answer['score']):
            points = np.array(box)
            print(points)
            xmin = points[0]
            # xmax = points[2]
            # ymin = points[1]
            # ymax = points[3]
            ymin = points[1]
            xmax = points[0] + points[2]
            ymax = points[1] + points[3]
            if score<=0.5:
                pass
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t\t<object>\n')
                xml.write('\t\t\t<coordinate>pixel</coordinate>\n')
                xml.write('\t\t\t<type>rectangle</type>\n')
                xml.write('\t\t\t<description>None</description>\n')
                xml.write('\t\t\t<possibleresult>\n')
                xml.write('\t\t\t\t<name>ship</name>\n')
                xml.write('\t\t\t</possibleresult>\n')
                xml.write('\t\t\t<points>\n')
                xml.write('\t\t\t\t<point>' + str(xmin) + ', ' + str(ymin) + '</point>\n')
                xml.write('\t\t\t\t<point>' + str(xmin) + ', ' + str(ymax) + '</point>\n')
                xml.write('\t\t\t\t<point>' + str(xmax) + ', ' + str(ymax) + '</point>\n')
                xml.write('\t\t\t\t<point>' + str(xmax) + ', ' + str(ymin) + '</point>\n')
                xml.write('\t\t\t\t<point>' + str(xmin) + ', ' + str(ymin) + '</point>\n')
                xml.write('\t\t\t</points>\n')
                xml.write('\t\t</object>\n')
            # print(multi['name'],xmin,ymin,xmax,ymax,label)
        xml.write('\t<objects>\n')
        xml.write('</annotation>')

# 5.复制图片到 VOC2007/JPEGImages/下

# for image_path in [image_path1, image_path2]:
#     image_files = glob(image_path + "*.jpg")
#     print("copy image files to VOC007/JPEGImages/")
#     for image in tqdm(image_files):
#         shutil.copy(image, saved_path + "JPEGImages/")

# #6.split files for txt
# txtsavepath = saved_path + "ImageSets/Main/"
# ftrainval = open(txtsavepath + '/trainval.txt', 'w')
# ftest = open(txtsavepath + '/test.txt', 'w')
# ftrain = open(txtsavepath + '/train.txt', 'w')
# fval = open(txtsavepath + '/val.txt', 'w')
# total_files = glob("VOC2007/Annotations/*.xml")
# total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# # test_filepath = ""
# for file in total_files:
#     ftrainval.write(file + "\n")
# # test
# # for file in os.listdir(test_filepath):
# #     ftest.write(file.split(".jpg")[0] + "\n")
# # split
# train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)
# # train
# for file in train_files:
#     ftrain.write(file + "\n")
# # val
# for file in val_files:
#     fval.write(file + "\n")

# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()

