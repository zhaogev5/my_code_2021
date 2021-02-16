import os  
import random  

xmlfilepath = '789/voc_xml'  
txtsavepath = 'yolo_try/Main'  
total_xml = os.listdir(xmlfilepath)  
  
num=len(total_xml)  
list=range(num)  
# tv=int(num)  
# # tr=int(tv*train_percent)  
# trainval= random.sample(list,tv)  
# train=random.sample(trainval,tr)  
  
# ftrainval = open('ImageSets/Main/trainval.txt', 'w')  
# ftest = open('ImageSets/Main/test.txt', 'w')  
ftrain = open('yolo_try/Main/train_1.txt', 'w')  
# fval = open('ImageSets/Main/val.txt', 'w')  
  
for i  in list:  
    name=total_xml[i][:-4]+'\n'  
    # if i in trainval:  
    ftrain.write(name)  
    #     if i in train:  
    #         ftrain.write(name)  
    #     else:  
    #         fval.write(name)  
    # else:  
    #     ftest.write(name)  
  
# ftrainval.close()  
ftrain.close()  
# fval.close()  
# ftest.close()