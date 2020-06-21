# DL2020-Image-Captioning

There are two possible ways to execute this project:
 - Run on the UPF cluster (Project)
 - Portable version that runs on command line with the models we generated (ImageCaptioning)

## Portable Version
LSTM : python3 ImageCaptioning.py -image [image name] -model [alexnet/squeeze/vgg/resnet18] -epoch [numberTrainedEpochs]

GRU  : python3 ImageCaptioning.py -image [image name] -model [alexnet/squeeze/vgg/resnet18] -epoch [numberTrainedEpochs] -rnn gru

## Whole project execution (UPF cluster)
This codes is made to run on the UPF cluster, to change the path (just the user uXXXXXX)
Change path on: 
 - line 330 / 332
 - line 425 / 427
 - line 511, 514 and 583 

Lines 423 / 442 used to divide the dataset on diferent folders, this process takes a while, so once you have done a first
run, it is highly recomended to comment this code. 

Line 603 -> change the model used to one of the following 
 - resnet18
 - resnet152 (too long to traing)
 - alexnet
 - vgg
 - inception (not tested)
 - squeeze
 - dense (not tested)

Line 618 -> change number of epochs

##

To see all models:
https://drive.google.com/drive/folders/1vxMxWjTeSJ2z_HjyAb9GwXp2gZPQ_Aem?usp=sharing

Add the dataset under a data/ folder on root directory. Link to dataset:
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
