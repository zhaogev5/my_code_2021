from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import xml.dom.minidom
import os
from PIL import Image
import xml.etree.ElementTree as ET

def as_num(x):
    y = '{:.5f}'.format(x)
    return(y)

def deal(xmlPath,path):
    files = os.listdir(path)  # 列出所有文件
    for file in files:
        filename = os.path.splitext(file)[0]  # 分割出文件名
        sufix = os.path.splitext(file)[1]  # 分割出后缀
        if sufix == '.xml':
            xmins = []
            ymins = []
            xmaxs = []
            ymaxs = []
            num, xmins, ymins, xmaxs, ymaxs = readtxt(file)
            dealpath = xmlPath + filename + ".xml"
            filename = filename + '.jpg'
            with open(dealpath, 'w') as f:
                writexml(dealpath, filename, num, xmins, ymins, xmaxs, ymaxs)

# 读取图片的高和宽写入xml
def dealwh(xmlPath,path):
    files = os.listdir(path)  # 列出所有文件
    for file in files:
        filename = os.path.splitext(file)[0]
        #print(filename)  # 分割出文件名
        sufix = os.path.splitext(file)[1]  # 分割出后缀
        if sufix == '.jpg':
            height, width = readsize(file)
            # print(height,width)
            dealpath = xmlPath + filename + ".xml"
            gxml(dealpath, height, width)

# 读取txt(xml)文件
def readtxt(p):
    p_file = txtPath + p
    tree=ET.parse(p_file)
    root = tree.getroot()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    num = 0
    for j in root.findall('objects'):
        for m in j.findall('object'):
            x = []
            y = []
            for k in m.find('points'):
                x.append(float(k.text.split(',')[0]))
                y.append(float(k.text.split(',')[1]))
            x_min=min(x)#max(min(x),0)
            y_min=min(y)#max(min(y),0)
            x_max=max(x)#min(max(x),i['width'])
            y_max=max(y)#min(max(y),i['height'])
            xmins.append(x_min)
            ymins.append(y_min)
            xmaxs.append(x_max)
            ymaxs.append(y_max)
            num = num + 1

        # print(num,xmins,ymins,xmaxs,ymaxs,names)
        return num, xmins, ymins, xmaxs, ymaxs

# 在xml文件中添加宽和高
def gxml(path, height, width):
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement
    heights = root.getElementsByTagName('height')[0]
    heights.firstChild.data = height
    # print(height)
 
    widths = root.getElementsByTagName('width')[0]
    widths.firstChild.data = width
    # print(width)
    with open(path, 'w') as f:
    # with open(xmlPath, 'w') as f:
        dom.writexml(f)
    return

# 创建xml文件
def writexml(path, filename,num, xmins, ymins, xmaxs, ymaxs, height='800', width='800'):
    node_root = Element('annotation')
 
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = "VOC2007"
 
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = "%s" % filename
 
    node_size = SubElement(node_root, "size")
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
 
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
 
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(num):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = '%s' % 'ship'
        node_name = SubElement(node_object, 'pose')
        node_name.text = '%s' % "unspecified"
        node_name = SubElement(node_object, 'truncated')
        node_name.text = '%s' % "0"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % xmins[i]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % ymins[i]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % xmaxs[i]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % ymaxs[i]
 
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    with open(path, 'wb') as f:
        f.write(xml)
    return

def readsize(p):
    p_file=imagePath+p
    img=Image.open(p_file)
    width = img.size[0]
    height = img.size[1]
    return height, width

if __name__ == "__main__":
    imagePath = ("./789/val2017/")
    txtPath = ("./789/val_xml/")
    xmlPath = ("./789/voc_val/")
    deal(xmlPath,txtPath)
    dealwh(xmlPath,imagePath)