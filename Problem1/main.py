import os
import os.path
from os.path import join
import argparse
import torch
import torch.nn as nn
from torch import optim
from dataloader import dataloader
from torch.utils.data import DataLoader
from model import model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import time

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--tsne', action = 'store_true')
	parser.add_argument('--train_img_path', type = str, default = '../hw2_data/p1_data/train_50')
	parser.add_argument('--val_img_path', type = str, default = '../hw2_data/p1_data/val_50')
	parser.add_argument('--gt_path', type = str, default = '../hw2_data/p1_data/val_gt.csv')
	parser.add_argument('--test_img_path', type = str)
	parser.add_argument('--pred_path', type = str, default = './test_pred.csv')
	return parser.parse_args()

def cal_hit(prediction, label):
	predict = torch.argmax(prediction, dim = 1, keepdim = True)
	return torch.sum(predict == label).item(), len(label)

def save_tsne(tsne_features, labels):
	tsne_features = np.array(tsne_features)
	labels = np.array(labels)
	tsne_features_fit = TSNE(n_components = 3).fit_transform(tsne_features, labels)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(tsne_features_fit[:, 0], tsne_features_fit[:, 1], tsne_features_fit[:, 2], c = labels, cmap = plt.cm.jet)
	plt.savefig('tsne.jpg')

def write_csv(filenames, predictions, pred_path):
	with open(pred_path, "w+") as f:
		first_row = ['image_id', 'label']
		f.write("%s,%s\n" %(first_row[0], first_row[1]))
		for index, filename in enumerate(filenames):
			f.write("%s,%s\n" %(filenames[index], predictions[index]))

def test_accuracy(pred_path, gt_path):
	pred_reader = pd.read_csv(pred_path).sort_values(['image_id'])
	gt_reader = pd.read_csv(gt_path).sort_values(['image_id'])
	pred = np.array(pred_reader['label'], dtype = np.int32)
	gt = np.array(gt_reader['label'], dtype = np.int32)
	print('Accuracy: {:.2f}%'.format(np.sum(pred == gt) / len(gt_reader) * 100))

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	if args.mode == 'train':
		batch_size = 48
		max_epoch = 20

		train_dataloader = dataloader(args.mode, args.train_img_path, args.val_img_path)
		train_data = DataLoader(train_dataloader, batch_size = batch_size, shuffle = True, num_workers = 6, pin_memory = True)
		valid_dataloader = dataloader('valid', args.train_img_path, args.val_img_path)
		valid_data = DataLoader(valid_dataloader, batch_size = batch_size, shuffle = False, num_workers = 6, pin_memory = True)
		print('loading model...')
		net = model()
		print(net)
		total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		net.cuda().float()

		save_path = './models/'
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		if args.load != -1:
			net.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))

		optimizer = optim.SGD(filter(lambda param : param.requires_grad, net.parameters()), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()
		best_loss = 100.0

		for epoch in range(args.load + 1, max_epoch):
			net.train()
			total_loss = 0
			hit_total = np.zeros(2)
			for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				prediction, tsne_feature = net(batch_images)
				loss = loss_func(prediction, batch_labels)
				total_loss += loss.item()
				hit_total += np.array(cal_hit(prediction, batch_labels.unsqueeze(-1)))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			avg_loss = total_loss / len(train_data)
			print('epoch:', epoch)
			print('train_avg_loss: {:.10f}'.format(avg_loss))
			print('train_accuracy: {:.2f}%'.format(hit_total[0] / hit_total[1] * 100))

			with torch.no_grad():
				net.eval()
				total_loss = 0
				hit_total = np.zeros(2)
				for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					prediction, tsne_feature = net(batch_images)
					loss = loss_func(prediction, batch_labels)
					total_loss += loss.item()
					hit_total += np.array(cal_hit(prediction, batch_labels.unsqueeze(-1)))
				avg_loss = total_loss / len(valid_data)
				print('epoch:', epoch)
				print('valid_avg_loss: {:.10f}'.format(avg_loss))
				print('valid_accuracy: {:.2f}%'.format(hit_total[0] / hit_total[1] * 100))
				if avg_loss < best_loss:
					best_loss = avg_loss
					torch.save(net.state_dict(), save_path + 'best.ckpt')
				torch.save(net.state_dict(), save_path + '{}.ckpt'.format(epoch))

	elif args.mode == 'valid':
		batch_size = 1
		valid_dataloader = dataloader(args.mode, args.train_img_path, args.val_img_path)
		valid_data = DataLoader(valid_dataloader, batch_size = batch_size, shuffle = False, num_workers = 6, pin_memory = True)

		print('loading model...')
		net = model()
		print(net)
		total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		net.cuda().float()

		save_path = './models/'
		net.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()

		with torch.no_grad():
			net.eval()
			total_loss = 0
			hit_total = np.zeros(2)
			tsne_features, predictions, labels, filenames = [], [], [], []
			for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
				batch_images, batch_labels, batch_filename = image.to(device), label.to(device), filename
				labels.append(batch_labels.cpu())
				prediction, tsne_feature = net(batch_images)
				loss = loss_func(prediction, batch_labels)
				total_loss += loss.item()
				hit_total += np.array(cal_hit(prediction, batch_labels.unsqueeze(-1)))
				tsne_features.append(tsne_feature.squeeze().cpu().numpy())
				filenames.append(batch_filename[0])
				predictions.append(torch.argmax(prediction.cpu()).item())
			avg_loss = total_loss / len(valid_data)
			print('epoch:', args.load)
			print('valid_avg_loss: {:.10f}'.format(avg_loss))
			print('valid_accuracy: {:.2f}%'.format(hit_total[0] / hit_total[1] * 100))
			if args.tsne:
				save_tsne(tsne_features, labels)
			write_csv(filenames, predictions, args.pred_path)
	elif args.mode == 'cal_ac':
		test_accuracy(args.pred_path, args.gt_path)

def test(args, device):
	start_time = time.time()
	test_dataloader = dataloader(args.mode, args.train_img_path, args.test_img_path)
	test_data = DataLoader(test_dataloader, batch_size = 1, shuffle = False, num_workers = 0)
	print('loading model...')
	net = model()
	print(net)
	total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	net.cuda().float()

	net.load_state_dict(torch.load('./Problem1/hw2_1.ckpt'))
	with torch.no_grad():
		net.eval()
		predictions, filenames = [], []
		for index, (image, filename) in enumerate(test_data):
			batch_images, batch_filename = image.to(device), filename
			prediction, tsne_feature = net(batch_images)
			filenames.append(batch_filename[0])
			predictions.append(torch.argmax(prediction.cpu()).item())
			if index % 100 == 0:
				print('progress: {:.1f}%'.format(index / len(test_data) * 100))
		print('progress: 100.0%')
		write_csv(filenames, predictions, join(args.pred_path, 'test_pred.csv'))
	print('Finished!')
	end_time = time.time()
	print('testing code finished in {:.1f} seconds'.format(end_time - start_time))

def resave(args):
	print('loading model...')
	net = model()
	print(net)
	total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	net.cuda().float()

	save_path = './models/'
	net.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))
	torch.save(net.state_dict(), save_path + '{}_new.ckpt'.format(args.load), _use_new_zipfile_serialization = False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.mode == 'resave':
		resave(args)
		exit()
	test(args, device) if args.mode == 'test' else train(args, device)
