# DL2020-Image-Captioning
LSTM : python3 ImageCaptioning.py -image [image name] -model [alexnet/squeeze/vgg/resnet18] -epoch [numberTrainedEpochs]
GRU  : python3 ImageCaptioning.py -image [image name] -model [alexnet/squeeze/vgg/resnet18] -epoch [numberTrainedEpochs] -rnn gru

This codes is made to run on the UPF cluster, to change the path (just the user uXXXXXX)
Change path on: 
 - line 330 / 332
 - line 425 / 427
 - line 511, 514 and 583 

Lines 423 / 442 used to divide the dataset on diferent folders, this process takes a while, so once you have done a first
run, it is highly recomended to comment this code. 

Line 603 -> change the model used
Line 618 -> change number of epochs
