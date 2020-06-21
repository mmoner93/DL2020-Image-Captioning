#Alexnet

import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Alexnet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Alexnet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        in_features = self.alexnet.classifier[6].in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.alexnet.classifier[6] = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.alexnet(images)
        # embed = Variable(embed.data)
        # embed = embed.view(embed.size(0), -1)
        # embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed
		
		
#DenseNet

import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class DenseNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(DenseNet, self).__init__()
        self.dense = models.densenet121(pretrained=True)
        self.linear = nn.Linear(self.dense.classifier.in_features, embedding_dim)
        self.dense.classifier = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.dense(images)
        return embed
		
		
#Inception

import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Inception(nn.Module):
    def __init__(self, embedding_dim=300):
        super(Inception, self).__init__()
        self.inception = models.inception_v3(pretrained=True)
        in_features = self.inception.fc.in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.inception.fc = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.inception(images)
        return embed
		
		
#Resnet
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Resnet(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Resnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.linear = nn.Linear(in_features, embedding_dim)
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet18(images)
        embed = Variable(embed.data)
        embed = embed.view(embed.size(0), -1)
        embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed
		
#Resnet152
		
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Resnet152(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Resnet152, self).__init__()
        self.resnet152 = models.resnet152(pretrained=True)
        self.linear = nn.Linear(self.resnet152.fc.in_features, embedding_dim)
        self.resnet152.fc = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet152(images)
        # embed = Variable(embed.data)
        # embed = embed.view(embed.size(0), -1)
        # embed = self.linear(embed)
        # embed = self.batch_norm(embed)
        return embed
		
#SqueezeNet
		
import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class SqueezeNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(SqueezeNet, self).__init__()
        self.squeeze = models.squeezenet1_1(pretrained=True)
        self.squeeze.num_classes = embedding_dim
        final_conv = nn.Conv2d(512, self.squeeze.num_classes, kernel_size=1)
        self.squeeze.classifier[1] = final_conv
        self.linear = self.squeeze.classifier[1]
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.squeeze(images)
        return embed		
		
#VGG

import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Vgg(nn.Module):
    def __init__(self, embedding_dim=300):
        super(Vgg, self).__init__()
        self.vgg = models.vgg11(pretrained=True)
        in_features = self.vgg.classifier[6].in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.vgg.classifier[6] = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.vgg(images)
        return embed
		
		
		
def get_cnn(architecture = 'resnet18', embedding_dim = 300, cnnn = 0):
	if architecture == 'resnet18':
		cnnn = Resnet(embedding_dim = embedding_dim)
	elif architecture == 'resnet152':
		cnnn = Resnet152(embedding_dim = embedding_dim)
	elif architecture == 'alexnet':
		cnnn = Alexnet(embedding_dim = embedding_dim) 
	elif architecture == 'vgg':
		cnnn = Vgg(embedding_dim = embedding_dim) 
	elif architecture == 'inception':
		cnnn = Inception(embedding_dim = embedding_dim) 
	elif architecture == 'squeeze':
		cnnn = SqueezeNet(embedding_dim = embedding_dim) 
	elif architecture == 'dense':
		cnnn = DenseNet(embedding_dim = embedding_dim) 
	return cnnn
		
		

print('models loaded')
#dataloader
	
import os
print('os')
import json
print('json')
import nltk
print('nltk')
import time
import torch
print('torch')
from PIL import Image

class DataLoader():
	def __init__(self, dir_path, vocab, transform):
		self.images = None
		self.captions_dict = None
		# self.data = None
		self.vocab = vocab
		self.transform = transform
		self.load_captions(dir_path)
		self.load_images(dir_path)
	
	def load_captions(self, captions_dir):
		caption_file = os.path.join(captions_dir, 'captions.txt')
		captions_dict = {}
		with open(caption_file) as f:
			for line in f:
				cur_dict = json.loads(line)
				for k, v in cur_dict.items():
					captions_dict[k] = v
		self.captions_dict = captions_dict
	
	def load_images(self, images_dir):
		files = os.listdir(images_dir)
		images = {}
		for cur_file in files:
			ext = cur_file.split('.')[1]
			if ext == 'jpg':
				images[cur_file] = self.transform(Image.open(os.path.join(images_dir, cur_file)))
		self.images = images
	
	def caption2ids(self, caption):
		vocab = self.vocab
		tokens = nltk.tokenize.word_tokenize(caption.lower())
		vec = []
		vec.append(vocab.get_id('<start>'))
		vec.extend([vocab.get_id(word) for word in tokens])
		vec.append(vocab.get_id('<end>'))
		return vec
	
	def gen_data(self):
		images = []
		captions = []
		for image_id, cur_captions in self.captions_dict.items():
			num_captions = len(cur_captions)
			images.extend([image_id] * num_captions)
			for caption in cur_captions:
				captions.append(self.caption2ids(caption))
		# self.data = images, captions
		data = images, captions
		return data

	def get_image(self, image_id):
		return self.images[image_id]
			
def shuffle_data(data, seed=0):
	images, captions = data
	shuffled_images = []
	shuffled_captions = []
	num_images = len(images)
	torch.manual_seed(seed)
	perm = list(torch.randperm(num_images))
	for i in range(num_images):
		shuffled_images.append(images[perm[i]])
		shuffled_captions.append(captions[perm[i]])
	return shuffled_images, shuffled_captions

print('dataloader complete')
		
	


path = "/shared/home/u124298/data"
results_path = "/shared/home/u124298/results"
images = path+"/Flicker8k_Dataset/"
		
			
import os
import json
import time
import numpy as np
from PIL import Image
from shutil import copyfile


def read_captions(filepath):
	captions_dict = {}
	with open(filepath) as f:
		for line in f:
			line_split = line.split(sep='\t', maxsplit=1)
			caption = line_split[1][:-1]
			id_image = line_split[0].split(sep='#')[0]
			if id_image not in captions_dict:
				captions_dict[id_image] = [caption]
			else:
				captions_dict[id_image].append(caption)
	return captions_dict

def get_ids(filepath):
	ids = []
	with open(filepath) as f:
		for line in f:
			ids.append(line[:-1])
	return ids

def copyfiles(dir_output, dir_input, ids):
	if not os.path.exists(dir_output):
		os.makedirs(dir_output)
	for cur_id in ids:
		path_input = os.path.join(dir_input, cur_id)
		path_output = os.path.join(dir_output, cur_id)
		copyfile(path_input, path_output)

def write_captions(dir_output, ids, captions_dict):
	output_path = os.path.join(dir_output, 'captions.txt')
	output = []
	for cur_id in ids:
		cur_dict = {cur_id: captions_dict[cur_id]}
		output.append(json.dumps(cur_dict))
		
	with open(output_path, mode='w') as f:
		f.write('\n'.join(output))

def segregate(dir_images, filepath_token, captions_path_input):
	print(dir_images)
	print(filepath_token)
	print(captions_path_input)
	dir_output = {'train': dir_models + '/train',
				  'dev'  : dir_models + '/dev',
				  'test' : dir_models + '/test'
				 }
	
	# id [caption1, caption2, ..]
	captions_dict = read_captions(filepath_token)
	
	# train, dev, test images mixture
	images = os.listdir(dir_images)
	
	# read ids
	ids_train = get_ids(captions_path_input['train'])
	ids_dev = get_ids(captions_path_input['dev'])
	ids_test = get_ids(captions_path_input['test'])
	
	# copy images to respective dirs
	copyfiles(dir_output['train'], dir_images, ids_train)
	copyfiles(dir_output['dev'], dir_images, ids_dev)
	copyfiles(dir_output['test'], dir_images, ids_test)
 
	
	# write id
	write_captions(dir_output['train'], ids_train, captions_dict)
	write_captions(dir_output['dev'], ids_dev, captions_dict)
	write_captions(dir_output['test'], ids_test, captions_dict)

def load_captions(captions_dir):
	caption_file = os.path.join(captions_dir, 'captions.txt')
	captions_dict = {}
	with open(caption_file) as f:
		for line in f:
			cur_dict = json.loads(line)
			for k, v in cur_dict.items():
				captions_dict[k] = v
	return captions_dict


#To segregate the dataSet - Only use once
if __name__ == '__main__':
	dir_images = '/shared/home/u124298/data/Flicker8k_Dataset'
	dir_text = '/shared/home/u124298/data/'
	dir_models = '/shared/home/u124298/data/models/'
	filename_token = 'Flickr8k.token.txt'
	filename_train = 'Flickr_8k.trainImages.txt'
	filename_dev = 'Flickr_8k.devImages.txt'
	filename_test = 'Flickr_8k.testImages.txt'
	filepath_token = dir_text + filename_token
	captions_path_input = {'train': dir_text + filename_train,
						   'dev': dir_text + filename_dev,
						   'test': dir_text + filename_test
						  }
	#/content/drive/Shared drives/DANI IVAN EDGAR MARÇAL/DEEPLEARNING/Ivan - Marçal/Deep_Learning2020/Final Project/data/flick8k/Flickr8k.token.txt
	tic = time.time()
	segregate(dir_images, filepath_token, captions_path_input)
	toc = time.time()
	print('time: %.2f mins' %((toc-tic)/60))
#dataSet segregated


print('preproces')
		
#vocabulari
import nltk

nltk.download('punkt')
		
		
import os
import nltk
import json
from collections import Counter
#from Preprocess import load_captions


class Vocabulary():
	def __init__(self, captions_dict, threshold):
		self.word2id = {}
		self.id2word = {}
		self.index = 0
		self.build(captions_dict, threshold)
	
	def add_word(self, word):
		if word not in self.word2id:
			self.word2id[word] = self.index
			self.id2word[self.index] = word
			self.index += 1
	
	def get_id(self, word):
		if word in self.word2id:
			return self.word2id[word]
		return self.word2id['<unk>']
	
	def get_word(self, index):
		return self.id2word[index]
	
	def build(self, captions_dict, threshold):
		counter = Counter()
		tokens = []
		for k, captions in captions_dict.items():
			for caption in captions:
				tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
		
		counter.update(tokens)
		
		words = [word for word, count in counter.items() if count >= threshold]
		
		self.add_word('<unk>')
		self.add_word('<start>')
		self.add_word('<end>')
		self.add_word('<pad>')
		
		for word in words:
			self.add_word(word)

	def get_sentence(self, ids_list):
		sent = ''
		for cur_id in ids_list:
			cur_word = self.id2word[cur_id.item()]
			sent += ' ' + cur_word
			if cur_word == '<end>':
				break
		return sent

if __name__ == '__main__':
	
	captions_dict = load_captions('/shared/home/u124298/data/models/train')
	vocab = Vocabulary(captions_dict, 5)
	#save vocab
	torch.save(vocab, '/shared/home/u124298/data/models/vocab'+'vocab.pkl')
	print(vocab.index)
		
		
		
print('vocab')
		
#Decoder

import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, caption):
        seq_length = len(caption) + 1
        embeds = self.word_embeddings(caption)
        embeds = torch.cat((features, embeds), 0)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        out = self.linear(lstm_out.view(seq_length, -1))
        return out

    def greedy(self, cnn_out, seq_len = 20):
        ip = cnn_out
        hidden = None
        ids_list = []
        for t in range(seq_len):
            lstm_out, hidden = self.lstm(ip.unsqueeze(1), hidden)
            # generating single word at a time
            linear_out = self.linear(lstm_out.squeeze(1))
            word_caption = linear_out.max(dim=1)[1]
            ids_list.append(word_caption)
            ip = self.word_embeddings(word_caption)
        return ids_list
		
print('decoder')
#Train


import os
import torch
import time
import pickle
import argparse
import torch.nn as nn
#from Decoder import RNN
#from utils import get_cnn
import matplotlib.pyplot as plt
#from Vocabulary import Vocabulary
from torchvision import transforms
from torch.autograd import Variable
#from Preprocess import load_captions
#from DataLoader import DataLoader, shuffle_data


train_dir = '/shared/home/u124298/data/models/train'
threshold = 5

captions_dict = load_captions(train_dir)
vocab = Vocabulary(captions_dict, threshold)

transform = transforms.Compose([transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
                ])

dataloader = DataLoader(train_dir, vocab, transform)
data = dataloader.gen_data()
print(train_dir + ' loaded')

# embedding_dim = 512
vocab_size = vocab.index
hidden_dim = 512
# learning_rate = 1e-3
model_name = 'squeeze'

cnn = get_cnn(architecture = model_name, embedding_dim = 512, cnnn = 0) #args.embedding_dim)
lstm = RNN(embedding_dim = 512, hidden_dim = 512, 
        vocab_size = vocab_size)   #first args.embedding_dim, second args.hidden_dim
print(type(cnn))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

cnn.to(device)
lstm.to(device)

criterion = nn.CrossEntropyLoss()
params = list(cnn.linear.parameters()) + list(lstm.parameters()) 
optimizer = torch.optim.Adam(params, lr = 1e-5) 
num_epochs = 101              ########################################

for epoch in range(num_epochs):
  shuffled_images, shuffled_captions = shuffle_data(data, seed = epoch)
  num_captions = len(shuffled_captions)
  loss_list = []
  tic = time.time()
  for i in range(num_captions):
    image_id = shuffled_images[i]
    image = dataloader.get_image(image_id)
    image = image.unsqueeze(0)
          
    image = Variable(image).to(device)
    caption = torch.cuda.LongTensor(shuffled_captions[i])

    caption_train = caption[:-1] # remove <end>
    cnn.zero_grad()
    lstm.zero_grad()
    
    cnn_out = cnn(image)
    lstm_out = lstm(cnn_out, caption_train)
    loss = criterion(lstm_out, caption)
    loss.backward()
    optimizer.step()
    loss_list.append(loss)
  toc = time.time()
  avg_loss = torch.mean(torch.Tensor(loss_list))	
  print('epoch %d avg_loss %f time %.2f mins' 
    %(epoch, avg_loss, (toc-tic)/60))		
  if epoch % 10 == 0:

    torch.save(cnn.state_dict(), results_path+ '/' + model_name +'iter_%d_cnn.pkl'%(epoch))
    torch.save(lstm.state_dict(), results_path+ '/' + model_name +'iter_%d_lstm.pkl'%(epoch))
    print('saving '+results_path+ '/' + model_name +'iter_%d_lstm.pkl'%(epoch))



print('train complete')

#Validation

import os
import torch
import time
import pickle
import argparse
from PIL import Image
import torch.nn as nn
#from utils import get_cnn
#from Decoder import RNN
#from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
#from DataLoader import DataLoader, shuffle_data


#############################
#train_dir = '/shared/home/u124298/data/models/train'
threshold = 5

captions_dict = load_captions(train_dir)
vocab = Vocabulary(captions_dict, threshold)

transform = transforms.Compose([transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
                ])

dataloader = DataLoader(train_dir, vocab, transform)
data = dataloader.gen_data()
print(train_dir + ' loaded')
#############################
embedding_dim = 512
vocab_size = vocab.index
print(vocab_size)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

hidden_dim = 512
criterion = nn.CrossEntropyLoss()
cnn = get_cnn(architecture = model_name, embedding_dim = embedding_dim, cnnn = 0)
lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
					vocab_size = vocab_size)

cnn.to(device)
lstm.to(device)

for iteration in range(0, num_epochs, 10):
	cnn_file = '/' + model_name +'iter_%d_cnn.pkl'%(iteration)
	lstm_file = '/' + model_name +'iter_%d_lstm.pkl'%(iteration)
	cnn.load_state_dict(torch.load(results_path + cnn_file))
	lstm.load_state_dict(torch.load(results_path + lstm_file))

	cnn.eval()
	lstm.eval()

	images, captions = data
	num_captions = len(captions)
	loss_list = []
	# tic = time.time()
	with torch.no_grad():
		for i in range(num_captions):
			image_id = images[i]
			image = dataloader.get_image(image_id)
			image = image.unsqueeze(0)
			image = Variable(image).to(device)							
			if torch.cuda.is_available():
				caption = torch.cuda.LongTensor(captions[i])
			else:
				caption = torch.LongTensor(captions[i])

			caption_train = caption[:-1] # remove <end>
			
			loss = criterion(lstm(cnn(image), caption_train), caption)
			
			loss_list.append(loss)
			# avg_loss = torch.mean(torch.Tensor(loss_list))
			# print('ex %d / %d avg_loss %f' %(i+1, num_captions, avg_loss), end='\r')
	# toc = time.time()
	avg_loss = torch.mean(torch.Tensor(loss_list))
	print('%d %f' %(iteration, avg_loss))

print('val complete')


#Test

import os
import torch
import pickle
import argparse
from PIL import Image
import torch.nn as nn
#from utils import get_cnn
#from Decoder import RNN
#from Vocabulary import Vocabulary
from torch.autograd import Variable
from torchvision import transforms
#from DataLoader import DataLoader, shuffle_data

transform = transforms.Compose([transforms.Resize((224, 224)), 
																transforms.ToTensor(),
																transforms.Normalize((0.5, 0.5, 0.5),
																											(0.5, 0.5, 0.5))
																])
dir_images = path + "/Flicker8k_Dataset/"
shuffled_images, shuffled_captions = shuffle_data(data, seed = 40)
image_id = shuffled_images[35]
image = dataloader.get_image(image_id)
image = image.unsqueeze(0)
			
#gpu_device
image = Variable(image).to(device)
image = transform(Image.open(dir_images + image_id))

embedding_dim = 512
vocab_size = vocab.index
hidden_dim = 512
cnn = get_cnn(architecture = model_name, embedding_dim = embedding_dim)
lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
						vocab_size = vocab_size)
# cnn.eval()

image = image.unsqueeze(0)

# image = Variable(image)
if torch.cuda.is_available():	
		cnn.to(device)
		lstm.to(device)
		image = Variable(image).to(device)
else:
	image = Variable(image)


cnn_file = '/' + model_name +'iter_%d_cnn.pkl'%(num_epochs - 1)
lstm_file = '/' + model_name +'iter_%d_lstm.pkl'%(num_epochs - 1)

cnn.load_state_dict(torch.load(results_path + cnn_file))
lstm.load_state_dict(torch.load(results_path + lstm_file))


cnn_out = cnn(image)
image = image[0,:,:,:]
plt.imshow(image.permute(1,2,0).squeeze().cpu().numpy())
ids_list = lstm.greedy(cnn_out)
print("Image : " + image_id)
print(vocab.get_sentence(ids_list))



print('test complete')
