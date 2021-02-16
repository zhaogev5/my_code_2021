import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
anno_list = os.listdir('./DIOR_anno')
import shutil

new_xml = './bridge'
new_pic = './bridge_pic'
old_pic_dir = './JPEGImages-trainval'
false_num = 0

for i in tqdm(anno_list):
    xml_file = os.path.join('./DIOR_anno',i)

    with open(xml_file,'r') as f:
        tree=ET.parse(f)
        root = tree.getroot()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name == 'bridge':
                old_pic = os.path.join(old_pic_dir,i.replace('xml','jpg'))
                new_xml_file = os.path.join(new_xml,i)
                new_pic_file = os.path.join(new_pic,i.replace('xml','jpg'))
                try:
                    shutil.copyfile(xml_file,new_xml_file)
                    shutil.copyfile(old_pic,new_pic_file)
                except:
                    print( i + 'is wrong')
                    false_num += 1

print(false_num)
