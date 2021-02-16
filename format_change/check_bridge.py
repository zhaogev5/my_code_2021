import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


new_xml = './bridge_anno_new'
anno_list = os.listdir('./bridge_anno_new')

for i in tqdm(anno_list):
    old_xml_file = os.path.join(new_xml,i)
    with open(old_xml_file,'r') as f:
        tree=ET.parse(f)
        root = tree.getroot()
        ob_node = root.findall('object')
        for node in ob_node:
            obj = node.find('name')
            if (obj.text != 'bridge'):
                # root.remove(node)
                print(obj.text)			#删除节点




