import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

transform = transforms.Compose([transforms.ToTensor()])

augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p = 0.2),
								   transforms.ToTensor(),
								   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
								  ])

hash_table = {(  0, 255, 255): 0,
			  (255, 255,   0): 1,
			  (255,   0, 255): 2,
			  (  0, 255,   0): 3,
			  (  0,   0, 255): 4,
			  (255, 255, 255): 5,
			  (  0,   0,   0): 6}

def pixel2label(image, table = hash_table):
	label = np.ones((512, 512, 1)).astype(np.int64) * 6
	for i in hash_table:
		label[np.where(np.all(image == i, axis = -1))] = hash_table[i]
	return label

class dataloader(Dataset):
	def __init__(self, mode, train_img_path, val_img_path):
		super(dataloader, self).__init__()
		self.mode = mode
		self.img_path = train_img_path if self.mode == 'train' else val_img_path
		self.img_list = sorted([name for name in listdir(self.img_path) if name.endswith('.jpg')])
		if self.mode != 'test':
			self.label_list = sorted([name for name in listdir(self.img_path) if name.endswith('.png')])

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		image = Image.open(join(self.img_path, self.img_list[index]))
		image = transform(image)
		if self.mode == 'test':
			return image, self.img_list[index]
		else:
			pixel = np.array(Image.open(join(self.img_path, self.label_list[index])))
			label = pixel2label(pixel)
			label = transforms.ToTensor()(label)
			return image, label, self.img_list[index]

if __name__ == '__main__':
	test = dataloader('train')
	test_data = DataLoader(test, batch_size = 1, shuffle = True)
	print(len(test_data))
	for index, (image, label) in enumerate(test_data):
		print(index, image.shape, label)
		break
