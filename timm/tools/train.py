import os
import torch
from .val import val
from tqdm import tqdm
import torch.nn as nn
from .test import test
import torch.optim as optim
from core.utils import show_accuracy
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from core.utils import cal_balance_weight
import torchvision.transforms as transforms
from core.dataset import make_labels_unlabels_datasets, Label_Datasets, Unlabel_Datasets


def train(args, model, train_datasets, val_datasets, train_config, val_config, test_config, datasets_config, config_name):
	project_name = os.path.split(args.config)[-1].split(os.path.splitext(os.path.split(args.config)[-1])[-1])[0]
	train_writer = SummaryWriter(comment=project_name)
	config_name = config_name.split('.')[0]
	print(config_name)
	root_dir = datasets_config['Root_Dir']
	unlabel_ratio = datasets_config['Unlabel_Ratio']
	obj_list = datasets_config['Obj_List']
	batch_size = train_config['Batch_Size']
	max_epoch = train_config['Max_Epoch']
	gpus = train_config['Gpus']
	lr = train_config['Lr']
	gamma = train_config['Gamma']
	milestones = train_config['Milestones']
	opt = train_config['Opt']
	val_interval = train_config['Val_Interval']
	loss = train_config['Loss']
	balance_weight = train_config['Balance_Weight']
	num_workers = datasets_config['Num_Workers']
	resume = train_config['Resume']
	input_size = datasets_config['Input_Size']
	mean = datasets_config['Mean']
	std = datasets_config['Std']
	supervise = train_config['Supervise']
	update_interval = train_config['Update_Interval']
	start_update = train_config['Start_Update']

	base_transforms = transforms.Compose([transforms.Resize((input_size, input_size)),
	                                      transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

	if resume == '':
		start_epoch = 1
	else:
		checkpoint = torch.load(resume)
		start_epoch = checkpoint['epoch'] + 1
		model.load_state_dict(checkpoint['model'].state_dict())

	if gpus is not None:
		is_use_gpu = True
		model = model.cuda()
	# model = nn.DataParallel(model, devices=gpus)
	else:
		is_use_gpu = False

	if balance_weight == 'Auto':
		balance_weight = cal_balance_weight(datasets_config, obj_list)
	elif isinstance(balance_weight, list):
		assert len(balance_weight) == len(obj_list), 'balance_weight is diffierent with obj_list'
	else:
		balance_weight = [1 for i in obj_list]
	balance_weight = torch.Tensor([i / max(balance_weight) for i in balance_weight]).cuda() if is_use_gpu \
		else torch.Tensor([i / max(balance_weight) for i in balance_weight])

	loss_func = eval('nn.' + loss)(balance_weight)
	optimizer = eval('optim.' + opt)(model.parameters(), lr)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

	if supervise == 'all':
		train_label_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,
		                                    num_workers=num_workers)
		train_num_iter = len(train_label_dataloader)
	elif supervise == 'half':
		label_img_list, unlabel_img_list = make_labels_unlabels_datasets(datasets_config)

	all_iter = 0

	for epoch in range(start_epoch, max_epoch + 1):
		model.train()
		loss_epoch = 0
		# scheduler.step()
		train_accuracy = {i: {'correct': 0, 'wrong': 0, 'other_wrong': 0} for i in obj_list}

		if supervise == 'half':
			train_label_datasets = Label_Datasets(label_img_list, base_transforms)
			train_label_dataloader = DataLoader(train_label_datasets, batch_size=batch_size,
			                                    shuffle=True, num_workers=num_workers)
			train_num_iter = len(train_label_dataloader)

			train_unlabel_datasets = Unlabel_Datasets(unlabel_img_list, base_transforms)

		with tqdm(total=train_num_iter) as train_bar:
			for iter, data in enumerate(train_label_dataloader):

				all_iter += 1
				model_input = data['img']
				labels = data['label']
				if is_use_gpu:
					for img in model_input:
						model_input[img] = model_input[img].cuda()
					labels = labels.cuda()
				model_output = model(model_input)

				_, argmaxs = torch.max(model_output, dim=1)
				for j in range(labels.shape[0]):
					label = obj_list[labels[j].item()]
					argmax = obj_list[argmaxs[j].item()]
					if argmax == label:
						train_accuracy[label]['correct'] += 1
					else:
						train_accuracy[label]['wrong'] += 1
						train_accuracy[argmax]['other_wrong'] += 1

				loss_batch = loss_func(model_output, labels)
				loss_epoch += loss_batch.item()
				optimizer.zero_grad()
				loss_batch.backward()
				optimizer.step()


				train_writer.add_scalar('loss', loss_batch.item(), global_step=all_iter)
				train_bar.set_description(
					'lr: {} Epoch: {}/{}. TRAIN. Iteration: {}/{}. Cls loss: {:.5f}. All_loss: {:.5f}'.format(
						optimizer.state_dict()['param_groups'][0]['lr'], epoch, max_epoch, iter + 1, train_num_iter + 1, loss_batch.item(), loss_epoch))
				train_bar.update(1)
			scheduler.step()
		# train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)

		if supervise == 'half' and epoch >= start_update and epoch % update_interval == 0:
			high_con_dict, low_con_dict = test(args, model, train_unlabel_datasets, test_config, datasets_config)
			label_img_list = label_img_list + high_con_dict
			unlabel_img_list = low_con_dict

		if epoch % val_interval == 0:
			val_accuracy = val(args, model, val_datasets, val_config, datasets_config)
			print('#####\tTrain\t#####')
			train_accuracy, train_table = show_accuracy(train_accuracy)
			print('#####\tVal\t#####')
			val_accuracy, val_table = show_accuracy(val_accuracy)
			train_writer.add_scalar('ALL_P', val_accuracy['ALL']['P'], global_step=epoch)
			train_writer.add_scalar('ALL_R', val_accuracy['ALL']['R'], global_step=epoch)
			if not os.path.exists(f'./work_dirs2/{config_name}'):
				os.mkdir(f'./work_dirs2/{config_name}')

			train_acc = train_accuracy['ALL']['P']
			val_acc = val_accuracy['ALL']['P']
			project_txt = open(f'./work_dirs2/{config_name}/{config_name}_results.txt', 'a+')
			project_txt.write(
				f'Epoch:{epoch}\tTrain_Acc: {train_acc}\tVal_Acc: {val_acc}\n{train_table}\n{val_table}\n\n')
			project_txt.close()
			# torch.save({'model': model, 'epoch': epoch}, f'./work_dirs/{project_name}/model_{epoch}.pth')
			torch.save({'model': model, 'epoch': epoch}, f'./work_dirs2/{config_name}/model_{epoch}.pth')
