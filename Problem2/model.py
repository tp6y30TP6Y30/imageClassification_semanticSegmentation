import torch.nn as nn
import torchvision.models as models
import torch
from vgg import vgg16, vgg16_bn

def freeze(model):
	for param in model.parameters():
		param.requires_grad = False

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, input):
		return input

class Conv_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
		super(Conv_Block, self).__init__()
		self.block = nn.Sequential(
						nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
						nn.BatchNorm2d(out_channels),
						nn.ReLU(True),
					 )
	def forward(self, input):
		feature = self.block(input)
		return feature

class Trans_Block(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
		super(Trans_Block, self).__init__()
		self.block = nn.Sequential(
						nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False),
						nn.BatchNorm2d(out_channels),
						nn.LeakyReLU(1, inplace = True),
					 )
	def forward(self, input):
		feature = self.block(input)
		return feature

class model(nn.Module):
	def __init__(self, improve):
		super(model, self).__init__()
		self.improve = improve
		if self.improve:
			# FCN 8s
			self.backbone = vgg16_bn(pretrained = True)
			self.backbone.avgpool = Identity()
			self.backbone.classifier = Identity()
			self.Upsample16x16 = nn.Upsample(scale_factor = 2)
			self.Upsample32x32 = nn.Sequential(
									Conv_Block(512, 512, 3, 1, 1),
									nn.Upsample(scale_factor = 2)
								 )
			self.magnitude = Conv_Block(256, 512, 5, 1, 2)
			self.Upsample64x64 = nn.Sequential(
									Conv_Block(512, 256, 3, 1, 1),
									Conv_Block(256, 128, 5, 1, 2),
									Conv_Block(128, 7, 7, 1, 3),
									nn.Upsample(scale_factor = 8)
								 )
		else:
			# FCN 32s
			self.backbone = vgg16(pretrained = True)
			self.backbone.avgpool = Identity()
			self.backbone.classifier = Identity()
			self.Upsample16x16 = nn.Sequential(
									Conv_Block(512, 512, 3, 1, 1),
									Conv_Block(512, 512, 3, 1, 1),
									Conv_Block(512, 256, 3, 1, 1),
									Trans_Block(256, 128, 4, 2, 1),
									Trans_Block(128, 128, 4, 2, 1),
									Trans_Block(128, 128, 4, 2, 1),
									Trans_Block(128, 128, 4, 2, 1),
									Trans_Block(128, 7, 4, 2, 1),
								 )

	def forward(self, input):
		if self.improve:
			_, features_64_32_16 = self.backbone(input)
			temp32x32 = self.Upsample16x16(features_64_32_16[2]) + features_64_32_16[1]
			temp64x64 = self.Upsample32x32(temp32x32) + self.magnitude(features_64_32_16[0])
			image = self.Upsample64x64(temp64x64)
		else:
			features, _ = self.backbone(input)
			features = features.reshape(-1, 512, 16, 16)
			image = self.Upsample16x16(features)
		return image
