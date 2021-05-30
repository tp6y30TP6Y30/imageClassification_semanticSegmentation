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
from PIL import Image
import time

hash_table = {0: (  0, 255, 255),
			  1: (255, 255,   0),
			  2: (255,   0, 255),
			  3: (  0, 255,   0),
			  4: (  0,   0, 255),
			  5: (255, 255, 255),
			  6: (  0,   0,   0)}

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	parser.add_argument('--improve', action = 'store_true')
	parser.add_argument('--train_img_path', type = str, default = '../hw2_data/p2_data/train/')
	parser.add_argument('--val_img_path', type = str, default = '../hw2_data/p2_data/validation/')
	parser.add_argument('--test_img_path', type = str)
	parser.add_argument('--pred_path', type = str, default = './val_img')
	return parser.parse_args()

def cal_hit(prediction, label):
	predict = torch.argmax(prediction, dim = 1, keepdim = True)
	return torch.sum(predict == label).item()

def predict2images(prediction_buffer):
	images = []
	for predict in prediction_buffer:
		predict = predict.permute(0, 2, 3, 1).to('cpu')
		for p in predict:
			image = np.zeros((512, 512, 3)).astype(np.uint8)
			for i in hash_table:
				image[np.where(torch.argmax(p, dim = -1) == i)] = hash_table[i]
			images.append(image)
	return images

def save_img(images, path, filenames):
	for index, image in enumerate(images):
		image = Image.fromarray(image)
		image.save(join(path, filenames[index][:filenames[index].find('_')] + '_mask.png'))

def run_miou_score(pred_path, label_path):
	import mean_iou_evaluate
	pred_path = pred_path
	label_path = label_path
	pred = mean_iou_evaluate.read_masks(pred_path)
	label = mean_iou_evaluate.read_masks(label_path)
	mean_iou_evaluate.mean_iou_score(pred, label)

def train(args, device):
	torch.multiprocessing.freeze_support()
	from tqdm import tqdm
	if args.mode == 'train':
		batch_size = 8
		max_epoch = 30

		train_dataloader = dataloader(args.mode, args.train_img_path, args.val_img_path)
		train_data = DataLoader(train_dataloader, batch_size = batch_size, shuffle = True, num_workers = 6, pin_memory = True)

		valid_dataloader = dataloader('valid', args.train_img_path, args.val_img_path)
		valid_data = DataLoader(valid_dataloader, batch_size = 1, shuffle = False, num_workers = 6, pin_memory = True)

		print('loading model...')
		net = model(args.improve)
		print(net)
		total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		net.cuda().float()

		model_path = './models_{}/'.format('improve' if args.improve else 'baseline')
		if not os.path.exists(model_path):
			os.mkdir(model_path)

		if args.load != -1:
			net.load_state_dict(torch.load(model_path + str(args.load) + '.ckpt'))

		optimizer = optim.SGD(filter(lambda param : param.requires_grad, net.parameters()), lr = 1e-3, weight_decay = 0.012, momentum = 0.9)
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()
		best_loss = 100.0

		for epoch in range(args.load + 1, max_epoch):
			net.train()
			total_loss = 0
			hit = 0

			for index, (image, label, filename) in enumerate(tqdm(train_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				prediction = net(batch_images)
				loss = loss_func(prediction, batch_labels.squeeze(1))
				total_loss += loss.item()
				hit += cal_hit(prediction, batch_labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			avg_loss = total_loss / len(train_data)
			print('epoch:', epoch)
			print('train_avg_loss: {:.10f}'.format(avg_loss))
			print('train_accuracy: {:.2f}%'.format(hit / (len(train_data) * batch_size * 512 * 512) * 100))

			with torch.no_grad():
				net.eval()
				total_loss = 0
				hit = 0
				prediction_buffer, filenames = [], []

				for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
					batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
					prediction = net(batch_images)
					prediction_buffer.append(prediction)
					filenames.append(batch_filenames[0])
					loss = loss_func(prediction, batch_labels.squeeze(1))
					total_loss += loss.item()
					hit += cal_hit(prediction, batch_labels)

				avg_loss = total_loss / len(valid_data)
				print('epoch:', epoch)
				print('valid_avg_loss: {:.10f}'.format(avg_loss))
				print('valid_accuracy: {:.2f}%'.format(hit / (len(valid_data) * 1 * 512 * 512) * 100))

				val_dir = args.pred_path + '_' + ('improve/' if args.improve else 'baseline/')
				if not os.path.exists(val_dir):
					os.mkdir(val_dir)
				save_path = val_dir + 'epoch_{}'.format(epoch)
				if not os.path.exists(save_path):
					os.mkdir(save_path)
				print('image predicting...')
				val_img = predict2images(prediction_buffer)
				print('image saving...')
				save_img(val_img, save_path, filenames)
				print('miou_score testing...')
				run_miou_score(save_path, args.val_img_path)

				if avg_loss < best_loss:
					best_loss = avg_loss
					torch.save(net.state_dict(), model_path + 'best.ckpt')

				torch.save(net.state_dict(), model_path + '{}.ckpt'.format(epoch))

	elif args.mode == 'valid':
		batch_size = 1
		valid_dataloader = dataloader(args.mode, args.train_img_path, args.val_img_path)
		valid_data = DataLoader(valid_dataloader, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)

		print('loading model...')
		net = model(args.improve)
		print(net)
		total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
		print("Total number of params = ", total_params)
		net.cuda().float()

		save_path = './models_{}/'.format('improve' if args.improve else 'baseline')
		net.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))
		loss_func = nn.CrossEntropyLoss()
		loss_func.cuda()

		with torch.no_grad():
			net.eval()
			total_loss = 0
			hit = 0
			prediction_buffer, filenames = [], []

			for index, (image, label, filename) in enumerate(tqdm(valid_data, ncols = 70)):
				batch_images, batch_labels, batch_filenames = image.to(device), label.to(device), filename
				prediction = net(batch_images)
				prediction_buffer.append(prediction)
				filenames.append(batch_filenames[0])
				loss = loss_func(prediction, batch_labels.squeeze(1))
				total_loss += loss.item()
				hit += cal_hit(prediction, batch_labels)

			avg_loss = total_loss / len(valid_data)
			print('epoch:', args.load)
			print('valid_avg_loss: {:.10f}'.format(avg_loss))
			print('valid_accuracy: {:.2f}%'.format(hit / (len(valid_data) * batch_size * 512 * 512) * 100))

			if not os.path.exists(args.pred_path):
				os.mkdir(args.pred_path)
			print('image predicting...')
			val_img = predict2images(prediction_buffer)
			print('image saving...')
			save_img(val_img, args.pred_path, filenames)
			print('miou_score testing...')
			run_miou_score(args.pred_path, args.val_img_path)
	elif args.mode == 'cal_miou':
		run_miou_score(args.pred_path, args.test_img_path)

def test(args, device):
	start_time = time.time()
	test_dataloader = dataloader(args.mode, args.train_img_path, args.test_img_path)
	test_data = DataLoader(test_dataloader, batch_size = 1, shuffle = False, num_workers = 0)
	print('loading model...')
	net = model(args.improve)
	print(net)
	total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	net.cuda().float()

	net.load_state_dict(torch.load('./Problem2/hw2_2_best.ckpt' if args.improve else './Problem2/hw2_2.ckpt'))
	with torch.no_grad():
		net.eval()
		predictions, filenames = [], []
		for index, (image, filename) in enumerate(test_data):
			batch_images, batch_filename = image.to(device), filename
			prediction = net(batch_images)
			predictions.append(prediction)
			filenames.append(batch_filename[0])
			if index % 100 == 0:
				print('progress: {:.1f}%'.format(index / len(test_data) * 100))
		print('progress: 100.0%')
		val_img = predict2images(predictions)
		save_img(val_img, args.pred_path, filenames)
	print('Finished!')
	end_time = time.time()
	print('testing code finished in {:.1f} seconds'.format(end_time - start_time))

def resave(args):
	print('loading model...')
	net = model(args.improve)
	print(net)
	total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total number of params = ", total_params)
	net.cuda().float()
	
	model_path = './models_{}/'.format('improve' if args.improve else 'baseline')
	save_path = './models_{}/'.format('improve' if args.improve else 'baseline')
	net.load_state_dict(torch.load(save_path + str(args.load) + '.ckpt'))
	torch.save(net.state_dict(), model_path + '{}_new.ckpt'.format(args.load), _use_new_zipfile_serialization = False)

if __name__ == '__main__':
	args = _parse_args()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	if args.mode == 'resave':
		resave(args)
		exit()
	test(args, device) if args.mode == 'test' else train(args, device)
