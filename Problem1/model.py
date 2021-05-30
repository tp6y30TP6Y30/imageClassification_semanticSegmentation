import torch.nn as nn
import torch
from vgg import vgg16_bn

def freeze(model):
	for param in model.parameters():
		param.requires_grad = False

def weights_init_uniform(m):
	classname = m.__class__.__name__
	# for every Linear layer in a model..
	if classname.find('Linear') != -1:
		# apply a uniform distribution to the weights and a bias=0
		m.weight.data.uniform_(0.0, 1.0)
		m.bias.data.fill_(0)

class model(nn.Module):
	def __init__(self):
		super(model, self).__init__()
		self.backbone = vgg16_bn(pretrained = True)
		self.backbone.classifier[-1] = nn.Linear(4096, 50)

	def forward(self, input):
		predict, tsne_feature = self.backbone(input)
		return predict, tsne_feature
