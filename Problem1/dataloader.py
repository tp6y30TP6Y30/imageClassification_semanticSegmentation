import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

transform = transforms.Compose([transforms.Resize((224, 224)),
								transforms.ToTensor(),
								transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
							   ])

augmentation = transforms.Compose([transforms.Resize((224, 224)),
								   transforms.RandomHorizontalFlip(p = 0.2),
								   transforms.ToTensor(),
								   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
								  ])

class dataloader(Dataset):
	def __init__(self, mode, train_img_path, val_img_path):
		super(dataloader, self).__init__()
		self.mode = mode
		self.img_path = train_img_path if self.mode == 'train' else val_img_path
		self.img_list = sorted(listdir(self.img_path))

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, index):
		image = Image.open(join(self.img_path, self.img_list[index]))
		image = augmentation(image) if self.mode == 'train' else transform(image)
		if self.mode == 'test':
			return image, self.img_list[index]
		else:
			label = np.array(self.img_list[index][:self.img_list[index].find('_')]).astype(np.int64)
			return image, label, self.img_list[index]

if __name__ == '__main__':
	test = dataloader('train')
	test_data = DataLoader(test, batch_size = 1, shuffle = False)
	for index, (image, label) in enumerate(test_data):
		print(index, image.shape, label)
		break
