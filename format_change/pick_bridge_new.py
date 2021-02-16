import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import shutil
from lxml.etree import Element, SubElement, tostring

new_xml = './bridge_anno_new'
old_xml = './bridge'
anno_list = os.listdir('./bridge')
pic_list = os.listdir('./bridge_pic')

for i in tqdm(pic_list):
    xml_name = i.split('.')[0] + '.xml'
    old_xml_file = os.path.join(old_xml,xml_name)
    with open(old_xml_file,'r') as f:
        tree=ET.parse(f)
        root = tree.getroot()
        ob_node = root.findall('object')
        for node in ob_node:
            obj = node.find('name')
            if (obj.text != 'bridge'):
                # root.remove(node)
                print(obj.text)			#删除节点
        # # # xml_string = tostring(root, pretty_print=True)
        # new_xml_file = os.path.join(new_xml,xml_name)
        # # # with open(new_xml_file,'w') as f1:
        # tree.write(new_xml_file)




