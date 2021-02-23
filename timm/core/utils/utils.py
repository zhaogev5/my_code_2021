import os
import torch
import random
import prettytable as pt
from torchvision import transforms


def randam_string(num):
	letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
	salt = ''
	for i in range(num):
		salt += random.choice(letters)
	return salt


def cal_balance_weight(dataset_config, class_list):
	root_dir = dataset_config['Root_Dir']
	sensors = dataset_config['Sensors']
	num_sample = [len(os.listdir(os.path.join(root_dir, 'train', sensors[0], i))) for i in class_list]
	num_all_sample = sum(num_sample)
	weights = [num_all_sample / i for i in num_sample]
	return weights


def show_accuracy(accuracy):
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
	print(tb)

	Acc.update({'ALL': {'correct': correct, 'wrong': wrong, 'other_wrong': other_wrong, 'P': AP, 'R': AR}})
	return Acc, tb
