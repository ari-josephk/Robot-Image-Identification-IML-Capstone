import os
from torchvision.transforms import Normalize, ToTensor
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pickle as pkl

'''
Robot_Image_Dataset
Dataset class for the robot arm image data, utilizing lazy-loading

Constants:
    IMAGE0-2_MEAN - Estimated mean of the RGB values for each image (I got these using a ~500 sample of images)
    IMAGE0-2_STD - Estimated standard dev of the RGB values for each image (I got these using a ~500 sample of images)
'''
class Robot_Image_Dataset(Dataset):
    IMAGE0_MEAN = [0.4004, 0.4188, 0.4351]
    IMAGE1_MEAN = [0.4721, 0.4910, 0.5038]
    IMAGE2_MEAN = [0.4398, 0.4835, 0.5226]
    IMAGE0_STD = [0.2126, 0.2001, 0.2009]
    IMAGE1_STD = [0.2415, 0.2257, 0.2287]
    IMAGE2_STD = [0.2484, 0.2311, 0.2330]

    def __init__(self, path, train=True, double=False):
        path = os.path.join(path, 'train' if train else 'test')

        self.train = train
        self.double = double

        self.pathX = os.path.join(path, 'X')
        self.pathY = os.path.join(path, 'Y')

        self.data_folders = os.listdir(self.pathX)

        #define our transformations
        self.make_tensor = ToTensor()
        self.img0_normalize = Normalize(self.IMAGE0_MEAN, self.IMAGE0_STD)
        self.img1_normalize = Normalize(self.IMAGE1_MEAN, self.IMAGE1_STD)
        self.img2_normalize = Normalize(self.IMAGE2_MEAN, self.IMAGE2_STD)

    #transform each of our images by converting to tensor and normalizing
    def transform_img0(self, img):
        img = self.make_tensor(img)
        if self.double:
            img = img.double()
        return self.img0_normalize(img)

    def transform_img1(self, img):
        img = self.make_tensor(img)
        if self.double:
            img = img.double()
        return self.img1_normalize(img)

    def transform_img2(self, img):
        img = self.make_tensor(img)
        if self.double:
            img = img.double()
        return self.img2_normalize(img)
    
    #transform depths into a tensor
    def transform_depths(self, depths):
        depths = self.make_tensor(depths).swapaxes(0,1)
        if self.double:
            depths = depths.double()
        return depths
    

    def __len__(self):
        return len(self.data_folders)

    '''
    Get item at index 

    Params:
        idx - index of item to get
    '''
    def __getitem__(self, idx):
        f = self.data_folders[idx]

        #load in images from directory
        img0 = self.transform_img0(Image.open(os.path.join(self.pathX, f, 'rgb', '0.png')))
        img1 = self.transform_img1(Image.open(os.path.join(self.pathX, f, 'rgb', '1.png')))
        img2 = self.transform_img2(Image.open(os.path.join(self.pathX, f, 'rgb', '2.png')))

        #load in the .npy file for depths
        depth = self.transform_depths(np.load(os.path.join(self.pathX, f, 'depth.npy')))

        field_id = pkl.load(open(os.path.join(self.pathX, f, 'field_id.pkl'), 'rb'))

        #if we are in the train set, load in the result vector of positions
        finger_positions = np.load(os.path.join(self.pathY, f + '.npy')) if self.train else np.array([])

        return (img0, img1, img2, depth, field_id), finger_positions
