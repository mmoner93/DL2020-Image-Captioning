import os
import torch
import pickle
import numpy as np
import argparse
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-image')
	parser.add_argument('-model')
	parser.add_argument('-epoch', type=int)
	parser.add_argument('-rnn')
	args = parser.parse_args()

print(args.image)
print(args.rnn)

class Vgg(nn.Module):
    def __init__(self, embedding_dim=300):
        super(Vgg, self).__init__()
        self.vgg = models.vgg11(pretrained=True)
        in_features = self.vgg.classifier[6].in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.vgg.classifier[6] = self.linear
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.vgg(images)
        return embed

class Resnet(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Resnet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.linear = nn.Linear(in_features, embedding_dim)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet18(images)
        embed = Variable(embed.data)
        embed = embed.view(embed.size(0), -1)
        embed = self.linear(embed)
        return embed

class Alexnet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(Alexnet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        in_features = self.alexnet.classifier[6].in_features
        self.linear = nn.Linear(in_features, embedding_dim)
        self.alexnet.classifier[6] = self.linear
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.alexnet(images)
        return embed

class SqueezeNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(SqueezeNet, self).__init__()
        self.squeeze = models.squeezenet1_1(pretrained=True)
        self.squeeze.num_classes = embedding_dim
        final_conv = nn.Conv2d(512, self.squeeze.num_classes, kernel_size=1)
        self.squeeze.classifier[1] = final_conv
        self.linear = self.squeeze.classifier[1]
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.squeeze(images)
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

class RNN_GRU(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim)
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

path = os.getcwd()
print(path)

transform = transforms.Compose([transforms.Resize((224, 224)), 
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5),
								(0.5, 0.5, 0.5))
								])


### Prepare the image ###
image = transform(Image.open(path + '/' + args.image))
image = image.unsqueeze(0)

### Start the Model ###
vocab = torch.load(path + '/vocab.pkl')
embedding_dim = 512
hidden_dim = 512
add_path = 'Lstm'
cnn = get_cnn(architecture = args.model, embedding_dim = 512, cnnn = 0) #args.embedding_dim)
lstm = RNN(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
						vocab_size = vocab.index)
if(args.rnn == 'gru'):
	lstm = RNN_GRU(embedding_dim = embedding_dim, hidden_dim = hidden_dim, 
						vocab_size = vocab.index)
	add_path = 'Gru'

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if torch.cuda.is_available():	
	cnn.to(device)
	lstm.to(device)
	image = Variable(image).to(device)
else:
	image = Variable(image)


cnn_file = path +'/' + add_path +'/' + args.model +'iter_%d_cnn.pkl'%(args.epoch)
lstm_file = path + '/' + add_path +'/' + args.model +'iter_%d_lstm.pkl'%(args.epoch)
if(args.rnn == 'gru'):
	lstm_file = path + '/' + add_path +'/' + args.model +'iter_%d_gru.pkl'%(args.epoch)

cnn.load_state_dict(torch.load(cnn_file, map_location=map_location))
lstm.load_state_dict(torch.load(lstm_file, map_location=map_location))

cnn_out = cnn(image)
image = image[0,:,:,:]
ids_list = lstm.greedy(cnn_out)
print(vocab.get_sentence(ids_list))