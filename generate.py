import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from robot_image_dataset import Robot_Image_Dataset
from robot_image_cnn import Robot_Image_CNN

'''
Train and test a CNN model for robot images, create an output CSV file for Kaggle

Constants:
BATCH_SIZE - Number of samples per batch - reduce this if GPU memory errors occur
NUM_EPOCH - How many epochs to run in the model. Generally higher = better with convergence
USE_IMG0-2 - Whether to include images 0-2 in the model training and testing
'''
BATCH_SIZE = 32
NUM_EPOCH = 10
USE_IMG0 = True
USE_IMG1 = True
USE_IMG2 = True

print('Found CUDA Version: {}'.format(torch.version.cuda))

outfile = 'submission.csv'

output_file = open(outfile, 'w')

titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']

#create dataloader for train set
dataset = Robot_Image_Dataset(os.path.join(os.path.dirname(__file__),'data_lazy'))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#create our model and train it
model = Robot_Image_CNN(use_img0=USE_IMG0, use_img1=USE_IMG1, use_img2=USE_IMG2)
model.train(data_loader, num_epoch=NUM_EPOCH)

#create dataloader for test set
dataset_test = Robot_Image_Dataset(os.path.join(os.path.dirname(__file__),'data_lazy'), train=False)
data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

#use model to get our predictions
file_ids, preds = model.test(data_loader_test, return_field_ids=True)

file_ids = file_ids
preds = preds.cpu().detach().numpy()

#create the CSV output file
df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))
