import torch
from tqdm import tqdm
from core.utils import show_accuracy
from torch.utils.data import DataLoader


def val(args, model, val_datasets, val_config, datasets_config):
	model.eval()

	gpus = val_config['Gpus']
	batch_size = val_config['Batch_Size']
	confusion_matrix = val_config['Confusion_Matrix']
	num_workers = datasets_config['Num_Workers']
	obj_list = datasets_config['Obj_List']

	if gpus is not None:
		is_use_gpu = True
		model = model.cuda()
	else:
		is_use_gpu = False

	val_dataloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	val_num_iter = len(val_dataloader)

	val_accuracy = {i: {'correct': 0, 'wrong': 0, 'other_wrong': 0} for i in obj_list}
	with tqdm(total=val_num_iter) as val_bar:
		for iter, data in enumerate(val_dataloader):
			model_input = data['img']
			label = data['label']
			if is_use_gpu:
				for img in model_input:
					model_input[img] = model_input[img].cuda()
				labels = label.cuda()

			model_output = model(model_input)
			_, argmaxs = torch.max(model_output, dim=1)
			for j in range(labels.shape[0]):
				label = obj_list[labels[j].item()]
				argmax = obj_list[argmaxs[j].item()]
				if argmax == label:
					val_accuracy[label]['correct'] += 1
				else:
					val_accuracy[label]['wrong'] += 1
					val_accuracy[argmax]['other_wrong'] += 1
			val_bar.set_description(
				'VAL. Iteration: {}/{}.'.format(iter + 1, val_num_iter))
			val_bar.update(1)

	return val_accuracy
