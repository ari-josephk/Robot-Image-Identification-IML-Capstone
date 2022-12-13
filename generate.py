import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from robot_image_dataset import Robot_Image_Dataset
from robot_image_cnn import Robot_Image_CNN

BATCH_SIZE = 64

print('Found CUDA Version: {}'.format(torch.version.cuda))

outfile = 'submission.csv'

output_file = open(outfile, 'w')

titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']

dataset = Robot_Image_Dataset(os.path.join(os.path.dirname(__file__),'data_lazy'))
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Robot_Image_CNN(use_img0=True, use_img1=True, use_img2=False)
model.train(data_loader)


dataset_test = Robot_Image_Dataset(os.path.join(os.path.dirname(__file__),'data_lazy'), train=False)
data_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

file_ids, preds = model.test(data_loader_test, return_field_ids=True)

file_ids = file_ids
preds = preds.cpu().detach().numpy()

df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))
