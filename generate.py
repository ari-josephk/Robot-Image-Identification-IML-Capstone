import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from robot_image_dataset import Robot_Image_Dataset
from robot_image_cnn import Robot_Image_CNN

print('Found CUDA Version: {}'.format(torch.version.cuda))

outfile = 'submission.csv'

output_file = open(outfile, 'w')

titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
         'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
preds = []

dataset = Robot_Image_Dataset(os.path.join(os.path.dirname(__file__),'data_lazy'))
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Robot_Image_CNN(use_img0=True, use_img1=True, use_img2=True)
model.train(1, data_loader)



"""
test_data = torch.load(os.path.join(os.getcwd(),'data','test','test','testX.pt'))
file_ids = test_data[-1]
rgb_data = test_data[0]

print(test_data[0].shape)


model.eval()

for i, data in enumerate(rgb_data):
    # Please remember to modify this loop, input and output based on your model/architecture
    output = model(data[:1, :, :, :].to('cuda'))
    preds.append(output[0].cpu().detach().numpy())

df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
df.columns = titles
df.to_csv(outfile, index = False)
print("Written to csv file {}".format(outfile))
"""
