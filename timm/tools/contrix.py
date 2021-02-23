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
from core.model import build_model
from core.utils import load_config
from core.dataset import build_datasets


def contrix():
    is_use_gpu = True
    config = load_config('./config/resnet50_All.yaml')
    datasets_config = config['Dataset']
    model_config = config['Model']
    save_dir = './try'
    model = build_model(model_config, datasets_config, save_dir)
    checkpoint = torch.load('./work_dirs/resnet50_All/No_Fusion/model_100.pth')
    model.load_state_dict(checkpoint['model'].state_dict())
    val_datasets = build_datasets(datasets_config, 'val')
    val_dataloader = DataLoader(val_datasets, batch_size=4, shuffle=True, num_workers=num_workers)
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
