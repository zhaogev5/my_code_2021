import argparse
import torch
from tqdm import tqdm
import torch.nn as nn
from core.utils import show_accuracy
from torch.utils.data import DataLoader
from core.model import build_model
from core.utils import load_config
from core.dataset import build_datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np
import prettytable as pt
def save_accuracy(accuracy,kind,fusion,K):
    Acc = {i: {'correct': accuracy[i]['correct'], 'wrong': accuracy[i]['wrong'],
               'other_wrong': accuracy[i]['other_wrong']} for i in accuracy}
    tb = pt.PrettyTable()
    tb.field_names = ['Class_Name', 'TP', 'FN', 'FP', 'P', 'R']
    correct = 0
    wrong = 0
    other_wrong = 0
    for i in accuracy:
        correct += accuracy[i]['correct']
        wrong += accuracy[i]['wrong']
        other_wrong += accuracy[i]['other_wrong']
        P = accuracy[i]['correct'] / (accuracy[i]['correct'] + accuracy[i]['wrong']) \
            if accuracy[i]['correct'] + accuracy[i]['wrong'] != 0 else 0
        R = accuracy[i]['correct'] / (accuracy[i]['correct'] + accuracy[i]['other_wrong']) \
            if accuracy[i]['correct'] + accuracy[i]['other_wrong'] != 0 else 0
        tb.add_row([i, accuracy[i]['correct'], accuracy[i]['wrong'], accuracy[i]['other_wrong'], P, R])
        Acc[i].update({'P': P, 'R': R})
    AP = correct / (correct + wrong) if correct + wrong != 0 else 0
    AR = correct / (correct + other_wrong) if correct + other_wrong != 0 else 0

    tb.add_row([' ', correct, wrong, other_wrong, AP, AR])
    with open('结果.txt','a+') as f:
        f.write(f"{kind}_{fusion}\n")
        f.write(f"Kappa:{K}\n")
        f.write(str(tb))
        f.write('\n')

    Acc.update({'ALL': {'correct': correct, 'wrong': wrong, 'other_wrong': other_wrong, 'P': AP, 'R': AR}})
    return Acc, tb

def contrix(args,kind,fusion,tongdao=''):
    is_use_gpu = True
    config = load_config(args.config)
    datasets_config = config['Dataset']
    model_config = config['Model']
    obj_list = datasets_config['Obj_List']
    save_dir = args.save_dir
    model = build_model(model_config, datasets_config, save_dir)
    model = model.cuda()
    model.eval()
    if tongdao != '':
        tongdao = '_'+tongdao 
    checkpoint = torch.load(f'./work_dirs/resnet50_{kind}/{fusion}_Fusion{tongdao}/model_100.pth')
    model.load_state_dict(checkpoint['model'].state_dict())
    val_datasets = build_datasets(datasets_config, 'val')
    val_dataloader = DataLoader(val_datasets, batch_size=4, shuffle=True, num_workers=10)
    val_num_iter = len(val_dataloader)
    val_accuracy = {i: {'correct': 0, 'wrong': 0, 'other_wrong': 0} for i in obj_list}
    GT_list = []
    val_list = []
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
                print(type(obj_list))
                GT_list.append(int(label))
                val_list.append(int(argmax))
                if argmax == label:
                    val_accuracy[label]['correct'] += 1
                else:
                    val_accuracy[label]['wrong'] += 1
                    val_accuracy[argmax]['other_wrong'] += 1
            val_bar.set_description(
                'VAL. Iteration: {}/{}.'.format(iter + 1, val_num_iter))
            val_bar.update(1)
    # print(GT_list)
    # print(val_list)
    c_m=confusion_matrix(GT_list, val_list)
    K = kappa(c_m)
    # print(K)
    # print(type(c_m))
    print(type(c_m))
    plot_confusion_matrix(c_m,obj_list,f'{kind}_{fusion}_fusion')
    plt.savefig(f'{kind}_{fusion}_fusion{tongdao}.png', format='png')
    return val_accuracy, K

def plot_confusion_matrix(cm, labels_name, title):
    print(type(labels_name))
    print(cm.shape)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.figure()
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)   # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    # plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):    #第几行
        for second_index in range(len(cm[first_index])):    #第几列
            plt.text(first_index-0.25, second_index, str(round(float(cm[second_index][first_index]),2)), fontsize=8)

def kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = np.sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

if __name__ == '__main__':
    
    fusion_mode = 'Decision' #'Decision'
    tongdao = '' #没有填空
    kinds = ['All','Armored','Ground','Plane','Tank','Truck']
    kinds = ['All']
    # kinds = ['Ship','Harbor'] #'All',
    # kinds = ['Plane','Tank','Truck']
    for kind in kinds:
        parser = argparse.ArgumentParser(description='train')
        parser.add_argument('--train', action='store_true', default=False, help='config file')
        parser.add_argument('--val', action='store_true', default=False, help='config file')
        parser.add_argument('--test', action='store_true', default=False, help='config file')
        parser.add_argument('-c', '--config', type=str, default=f'./config/resnet50_{kind}.yaml', help='config file')
        parser.add_argument('--save_dir', type=str, default=f'{fusion_mode}_Fusion')
        args = parser.parse_args()

        val_acc,K=contrix(args,kind,fusion_mode,tongdao)
        show_accuracy(val_acc)
        save_accuracy(val_acc,kind,fusion_mode,K)
    

