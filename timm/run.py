import os
import argparse
from tools import train, val, test
from core.model import build_model
from core.dataset import build_datasets
from core.utils import load_config,show_accuracy
import torch




def runner(args):
    config_name = args.config.split('config/')[1]
    print(config_name)
    config = load_config(args.config)
    train_config = config['Train']
    val_config = config['Val']
    test_config = config['Test']
    datasets_config = config['Dataset']
    model_config = config['Model']
    # save_dir = args.save_dir

    model = build_model(model_config, datasets_config)
    # checkpoint = torch.load('./work_dirs/resnet50_All/Decision_Fusion/model_100.pth')
    # model.load_state_dict(checkpoint['model'].state_dict())
    train_datasets = build_datasets(datasets_config, 'train')
    val_datasets = build_datasets(datasets_config, 'val')
    # test_datasets = build_datasets(datasets_config, 'test')

    if args.train:
        train(args, model, train_datasets, val_datasets, train_config, val_config, test_config, datasets_config, config_name)
    if args.val:
        val_acc = val(args, model, val_datasets, val_config, datasets_config)
        show_accuracy(val_acc)
    if args.test:
        test(args, model, test_datasets, test_config, datasets_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--train', action='store_true', default=True, help='config file')
    parser.add_argument('--val', action='store_true', default=False, help='config file')
    parser.add_argument('--test', action='store_true', default=False, help='config file')
    parser.add_argument('-c', '--config', type=str, default='./config/resnet50_All_Decision.yaml', help='config file')
    # parser.add_argument('--save_dir', type=str, default='No_Fusion')
    args = parser.parse_args()
    runner(args)
