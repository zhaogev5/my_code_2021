import os
import json


IMAGES_DICT = ['file_name', 'height', 'width', 'id']
## {'license': 1, 'file_name': '000000516316.jpg', 'coco_url': '',
## 'height': 480, 'width': 640, 'date_captured': '2013-11-18 18:15:05',
## 'flickr_url': '', 'id': 516316}

ANNOTATIONS_DICT = ['image_id', 'iscrowd', 'area', 'bbox', 'category_id', 'id']
CATEGORIES_DICT = ['id', 'name']
## {'supercategory': 'person', 'id': 1, 'name': 'person'}
## {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}

DIOR_CATEGORIES = ["Boeing737", "Boeing747", "Boeing777", "Boeing787", "A220", "A321", "A330", "A350", "ARJ21", "other"]

def load_image(path):
	# img = cv2.imread(path)
	return 2048,2048#img.shape[0], img.shape[1]

def save_json(dict, path):
	print('SAVE_JSON...')
	with open(path, 'w') as f:
		json.dump(dict, f)
	print('SUCCESSFUL_SAVE_JSON:', path)

def generate_categories_dict(category):
	print('GENERATE_CATEGORIES_DICT...')
	return [{CATEGORIES_DICT[0]: category.index(x) + 1, CATEGORIES_DICT[1]: x} for x in category]


def generate_images_dict(imagelist, image_path):
	print('GENERATE_IMAGES_DICT...')
	images_dict = []
	with tqdm(total=len(imagelist)) as load_bar:
		for x in imagelist:
			dict = {IMAGES_DICT[0]: x, IMAGES_DICT[1]: load_image(image_path + x)[0], \
			        IMAGES_DICT[2]: load_image(image_path + x)[1], IMAGES_DICT[3]: len(images_dict)}
			load_bar.update(1)
			images_dict.append(dict)
	return images_dict



if __name__=='__main__':
	image_path=''
	save_path='test.json'
	categories_dict = generate_categories_dict(DIOR_CATEGORIES)

	imgname = os.listdir(image_path)
	images_dict = generate_images_dict(imgname, image_path)
	coco_dict={'images':images_dict,'annotations':[],'categories':categories_dict}
	save_json(coco_dict,save_path)