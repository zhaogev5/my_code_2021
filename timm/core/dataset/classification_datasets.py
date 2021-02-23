import os
import random
from PIL import Image
from torch.utils.data import Dataset


def label2unlabel(unlabel_img_list):
	for i in unlabel_img_list:
		i['label'] = None
	return unlabel_img_list


def make_labels_unlabels_datasets(dataset_config):
	set_name = 'train'
	root_dir = dataset_config['Root_Dir']
	sensors = dataset_config['Sensors']
	unlabel_ratio = dataset_config['Unlabel_Ratio']
	obj_list = dataset_config['Obj_List']

	img_list = {}
	# for sensor in self.sensors:
	for obj in obj_list:
		ever_sensor_img = {sensor: os.listdir(os.path.join(root_dir, set_name, sensor, obj)) for
		                   sensor in sensors}
		assert \
			False not in [ever_sensor_img[sensors[0]] == ever_sensor_img[i] for i in ever_sensor_img], \
			'diffierent sensors have diffierent images in train_dataset or val_dataset'
		ever_obj_img = []
		for img in os.listdir(os.path.join(root_dir, set_name, sensors[0], obj)):
			ever_obj_img.append({sensor: {
				'img_path': os.path.join(root_dir, set_name, sensor, obj, img),
				'img_name': img, 'label': obj_list.index(obj)} for sensor in sensors})
		img_list.update({obj: ever_obj_img})

	label_num = {i: int(len(img_list[i]) * (1 - unlabel_ratio)) for i in img_list}

	label_img_list = []
	unlabel_img_list = []
	for i in obj_list:
		label_img_list += img_list[i][:label_num[i]]
		unlabel_img_list += img_list[i][label_num[i]:]
	# unlabel_img_list = label2unlabel(unlabel_img_list)
	return label_img_list, unlabel_img_list


class Classification_Datasets(Dataset):
	"""docstring for Classification_Datasets"""

	def __init__(self, root_dir, set_name, sensors, unlabel_ratio, transforms):
		super(Classification_Datasets, self).__init__()
		self.root_dir = root_dir
		self.set_name = set_name
		self.sensors = sensors
		self.unlabel_ratio = unlabel_ratio if set_name == 'train' else 0
		self.transforms = transforms
		if not isinstance(self.sensors, list):
			if self.set_name != 'test':
				self.obj_list = os.listdir(os.path.join(self.root_dir, self.set_name))
				self.img_list = {i: [{'img_path': os.path.join(self.root_dir, self.set_name, i, j),
				                      'label': self.obj_list.index(i), 'img_name': j}
				                     for j in os.listdir(os.path.join(self.root_dir, self.set_name, i))] for i in
				                 self.obj_list}

				self.label_num = {i: int(len(self.img_list[i]) * (1 - self.unlabel_ratio)) for i in self.img_list}

				self.label_img_list = []
				self.unlabel_img_list = []
				for i in self.obj_list:
					self.label_img_list += self.img_list[i][:self.label_num[i]]
					self.unlabel_img_list += self.img_list[i][self.label_num[i]:]
			# self.unlabel_img_list = label2unlabel(self.unlabel_img_list)
			else:
				self.img_list = [
					{'img_path': os.path.join(self.root_dir, self.set_name, i), 'img_name': i, 'label': None}
					for i in os.listdir(os.path.join(self.root_dir, self.set_name))]
				self.label_img_list = []
				self.unlabel_img_list = self.img_list

		else:
			if self.set_name != 'test':
				self.obj_list = os.listdir(os.path.join(self.root_dir, self.set_name, self.sensors[0]))
				self.img_list = {}
				# for sensor in self.sensors:
				for obj in self.obj_list:
					ever_sensor_img = {sensor: os.listdir(os.path.join(self.root_dir, self.set_name, sensor, obj)) for
					                   sensor in self.sensors}
					assert \
						False not in [ever_sensor_img[self.sensors[0]] == ever_sensor_img[i] for i in ever_sensor_img], \
						'diffierent sensors have diffierent images in train_dataset or val_dataset'
					ever_obj_img = []
					for img in os.listdir(os.path.join(self.root_dir, self.set_name, self.sensors[0], obj)):
						ever_obj_img.append({sensor: {
							'img_path': os.path.join(self.root_dir, self.set_name, sensor, obj, img),
							'img_name': img, 'label': self.obj_list.index(obj)} for sensor in self.sensors})
					self.img_list.update({obj: ever_obj_img})

				self.label_num = {i: int(len(self.img_list[i]) * (1 - self.unlabel_ratio)) for i in self.img_list}

				self.label_img_list = []
				self.unlabel_img_list = []
				for i in self.obj_list:
					self.label_img_list += self.img_list[i][:self.label_num[i]]
					self.unlabel_img_list += self.img_list[i][self.label_num[i]:]
			# self.unlabel_img_list = label2unlabel(self.unlabel_img_list)
			else:
				ever_sensor_img = {sensor: os.listdir(os.path.join(self.root_dir, self.set_name, sensor)) for
				                   sensor in self.sensors}
				assert \
					False not in [ever_sensor_img[self.sensors[0]] == ever_sensor_img[i] for i in ever_sensor_img], \
					'diffierent sensors have diffierent images in test_dataset'

				self.img_list = []
				for img in os.listdir(os.path.join(self.root_dir, self.set_name, self.sensors[0])):
					self.img_list.append({sensor: {
						'img_path': os.path.join(self.root_dir, self.set_name, sensor, img),
						'img_name': img, 'label': None} for sensor in self.sensors})
				self.label_img_list = []
				self.unlabel_img_list = self.img_list

	def __getitem__(self, index):
		img_dict = {}
		if self.set_name != 'test':
			for sensor in self.sensors:
				label_img = Image.open(self.label_img_list[index][sensor]['img_path'])
				if self.transforms is not None:
					label_img = self.transforms(label_img)
				img_dict.update({sensor: label_img})
				label = self.label_img_list[index][sensor]['label']
			img_data = {'img': img_dict, 'label': label}
		else:
			for sensor in self.sensors:
				unlabel_img = Image.open(self.unlabel_img_list[index][sensor]['img_path'])
				if self.transforms is not None:
					unlabel_img = self.transforms(unlabel_img)
				img_dict.update({sensor: unlabel_img})
			img_data = {'img': img_dict, 'label': None}
		return img_data

	def __len__(self):
		return len(self.label_img_list) if self.set_name != 'test' else len(self.unlabel_img_list)


class Label_Datasets(Dataset):
	"""docstring for Label_Datasets"""

	def __init__(self, images_path, transforms):
		super(Label_Datasets, self).__init__()
		self.images_path = images_path
		self.transforms = transforms
		self.sensors = [i for i in images_path[0]]

	def __getitem__(self, index):
		img_dict = {}
		for sensor in self.sensors:
			label_img = Image.open(self.images_path[index][sensor]['img_path'])
			if self.transforms is not None:
				label_img = self.transforms(label_img)
			img_dict.update({sensor: label_img})
			label = self.images_path[index][sensor]['label']
		img_data = {'img_path': self.images_path[index], 'img': img_dict, 'label': label}
		return img_data

	def __len__(self):
		return len(self.images_path)


class Unlabel_Datasets(Dataset):
	"""docstring for Unlabel_Datasets"""

	def __init__(self, images_path, transforms):
		super(Unlabel_Datasets, self).__init__()
		self.images_path = images_path
		self.transforms = transforms
		self.sensors = [i for i in images_path[0]]

	def __getitem__(self, index):
		img_dict = {}
		for sensor in self.sensors:
			unlabel_img = Image.open(self.images_path[index][sensor]['img_path'])
			if self.transforms is not None:
				unlabel_img = self.transforms(unlabel_img)
			img_dict.update({sensor: unlabel_img})
		img_data = {'img_path': self.images_path[index], 'img': img_dict}
		return img_data

	def __len__(self):
		return len(self.images_path)
