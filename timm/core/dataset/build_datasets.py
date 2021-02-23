from .classification_datasets import Classification_Datasets, make_labels_unlabels_datasets
import torchvision.transforms as transforms


def build_datasets(datasets_config, set_name, transform=None):
	root_dir = datasets_config['Root_Dir']
	sensors = datasets_config['Sensors']
	unlabel_ratio = datasets_config['Unlabel_Ratio']
	input_size = datasets_config['Input_Size']
	mean = datasets_config['Mean']
	std = datasets_config['Std']
	base_transforms = [transforms.Resize((input_size, input_size)),
	                   transforms.ToTensor(),
	                   transforms.Normalize(mean=mean, std=std)]
	all_transforms = [transform] + base_transforms if transform is not None else base_transforms
	all_transforms = transforms.Compose(all_transforms)
	datasets = Classification_Datasets(root_dir, set_name, sensors, unlabel_ratio, all_transforms)
	return datasets
