#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:55:14 2019
@author: chenhonghu
"""
from xml.dom.minidom import Document
import os
import os.path
from PIL import Image
import cv2
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
xml_path='answer/test_1_4/'
# 加载文件名称
anno_result = pd.read_json(open("answer/libra_800_800.json", "r"))
file_name_list = list(set(anno_result['image_id']))
print(file_name_list)
for file_name in tqdm(file_name_list):
  savename=os.path.join(xml_path,str(file_name)+'.xml')     
  #print(savename)
  doc=Document()
  annotation=doc.createElement('annotation')
  doc.appendChild(annotation)
# source部分
  source=doc.createElement('source')
  annotation.appendChild(source)
  filename=doc.createElement('filename')
  source.appendChild(filename)
  imgname = str(file_name)+'.tif'
  filename_txt=doc.createTextNode(imgname)
  filename.appendChild(filename_txt)

  origin=doc.createElement('origin')
  source.appendChild(origin)
  origin_txt = doc.createTextNode('GF2/GF3')
  origin.appendChild(origin_txt)
#research部分
  research=doc.createElement('research')
  annotation.appendChild(research)  
  version=doc.createElement('version')
  research.appendChild(version)
  version_txt = doc.createTextNode('4.0')
  version.appendChild(version_txt)

  provider=doc.createElement('provider')
  research.appendChild(provider)
  provider_txt = doc.createTextNode('xidian university')
  provider.appendChild(provider_txt)

  author=doc.createElement('author')
  research.appendChild(author)
  author_txt = doc.createTextNode('ymhj')
  author.appendChild(author_txt)

  pluginname=doc.createElement('pluginname')
  research.appendChild(pluginname)
  pluginname_txt = doc.createTextNode('Ship Detection in SAR Images')
  pluginname.appendChild(pluginname_txt)

  pluginclass=doc.createElement('pluginclass')
  research.appendChild(pluginclass)
  pluginclass_txt = doc.createTextNode('Detection')
  pluginclass.appendChild(pluginclass_txt)

  time=doc.createElement('time')
  research.appendChild(time)
  time_txt = doc.createTextNode('2020-07-2020-11')
  time.appendChild(time_txt)
#object
  answer = anno_result[anno_result['image_id'] == file_name]
  object_1=doc.createElement('objects') #表示第一层次
  annotation.appendChild(object_1)  
  for box, score in zip(answer['bbox'], answer['score']):
      points = np.array(box)
      xmin = points[0]
      ymin = points[1]
      xmax = points[0] + points[2]
      ymax = points[1] + points[3]
      if score<=0.5:
          pass
      else:
           # 创建point
          object_2=doc.createElement('object') #表示第二层次
          object_1.appendChild(object_2)
          coordinate=doc.createElement('coordinate')
          object_2.appendChild(coordinate)
          coordinate_txt = doc.createTextNode('pixel')
          coordinate.appendChild(coordinate_txt)

          type_1=doc.createElement('type')
          object_2.appendChild(type_1)
          type_txt = doc.createTextNode('rectangle')
          type_1.appendChild(type_txt)

          description=doc.createElement('description')
          object_2.appendChild(description)
          description_txt = doc.createTextNode('None')
          description.appendChild(description_txt)

          possibleresult=doc.createElement('possibleresult')
          object_2.appendChild(possibleresult)
          name=doc.createElement('name')
          possibleresult.appendChild(name)
          name_txt = doc.createTextNode('ship')
          name.appendChild(name_txt)

          probablity=doc.createElement('probability')
          possibleresult.appendChild(probablity)
          probablity_txt = doc.createTextNode(str(score))  #概率
          probablity.appendChild(probablity_txt)
          #点
          points=doc.createElement('points')
          object_2.appendChild(points)

          point1=doc.createElement('point')
          dot1 = str(xmin) +', ' +str(ymin)
          points.appendChild(point1)
          point_txt = doc.createTextNode(dot1)  #概率
          point1.appendChild(point_txt)


          point2=doc.createElement('point')
          dot2 = str(xmin) +', ' +str(ymax)
          points.appendChild(point2)
          point_txt = doc.createTextNode(dot2)  #概率
          point2.appendChild(point_txt)

          point3=doc.createElement('point')
          dot3 = str(xmax) +', ' +str(ymax)
          points.appendChild(point3)
          point_txt = doc.createTextNode(dot3)  #概率
          point3.appendChild(point_txt)

          point4=doc.createElement('point')
          dot4 = str(xmax) +', ' +str(ymin)
          points.appendChild(point4)
          point_txt = doc.createTextNode(dot4)  #概率
          point4.appendChild(point_txt)

          point1=doc.createElement('point')
          dot1 = str(xmin) +', ' +str(ymin)
          points.appendChild(point1)
          point_txt = doc.createTextNode(dot1)  #概率
          point1.appendChild(point_txt)
  with open(savename,'wb') as f:
      f.write(doc.toprettyxml(indent='\t',encoding='utf-8'))
  f.close()

                  