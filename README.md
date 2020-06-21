# DL2020-Image-Captioning

There are two possible ways to execute this project:
 - Run on the UPF cluster (Project)
 - Portable version that runs on command line with the models we generated (ImageCaptioning)

## Portable Version
To execute this, download the ImageCaptioning folder, then you only need to execute the following commands depending on 
the model you want to test. If you want to test more models, see links below to download them.
**LSTM** : 

`python3 ImageCaptioning.py -image [image name] -model [squeeze/resnet18] -epoch [numberTrainedEpochs]`

**GRU**  : 

`python3 ImageCaptioning.py -image [image name] -model [squeeze/resnet18] -epoch [numberTrainedEpochs] -rnn gru`

## Whole project execution (UPF cluster)
This codes is made to run on the UPF cluster, to change the path (just the user uXXXXXX).

To run the provided code on the **Project folder** you must also download de .sh files so it can be executed.
This code will run all steps of the project: preparing the dataset, generating a vocabulary, training, validation and testing.
It has been slightly modified so any excess code was removed, if any errors pop up please contact us at : 
ivan.martinez01@estudiant.upf.edu or marcal.moner01@estudiant.upf.edu
Change path on: 
 - line 330 / 332
 - line 425 / 427
 - line 511, 514 and 583 

Lines 423 / 442 used to divide the dataset on diferent folders, this process takes a while, so once you have done a first
run, it is highly recomended to comment this code. Also keep in mind that the Flick8k Dataset must be on the right designed 
path.

Line 603 -> change the model used to one of the following 
 - resnet18
 - resnet152 (too long to traing)
 - alexnet
 - vgg
 - inception (not tested)
 - squeeze
 - dense (not tested)

Line 618 -> change number of epochs (default 100 epochs (101 so last iter is also saved)).

##

To see all models:
https://drive.google.com/drive/folders/1vxMxWjTeSJ2z_HjyAb9GwXp2gZPQ_Aem?usp=sharing

Add the dataset under a data/ folder on root directory. Link to dataset:
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
