import os
from torchvision.transforms import Normalize, ToTensor
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pickle as pkl

class Robot_Image_Dataset(Dataset):
    IMAGE0_MEAN = [0.4004, 0.4188, 0.4351]
    IMAGE1_MEAN = [0.4721, 0.4910, 0.5038]
    IMAGE2_MEAN = [0.4398, 0.4835, 0.5226]
    IMAGE0_STD = [0.2126, 0.2001, 0.2009]
    IMAGE1_STD = [0.2415, 0.2257, 0.2287]
    IMAGE2_STD = [0.2484, 0.2311, 0.2330]
    DEPTH_NORMALIZER = 1000

    def __init__(self, path, train=True):
        path = os.path.join(path, 'train' if train else 'test')

        self.pathX = os.path.join(path, 'X')
        self.pathY = os.path.join(path, 'Y')

        self.data_folders = os.listdir(self.pathX)

        self.make_tensor = ToTensor()
        self.img0_normalize = Normalize(self.IMAGE0_MEAN, self.IMAGE0_STD)
        self.img1_normalize = Normalize(self.IMAGE1_MEAN, self.IMAGE1_STD)
        self.img2_normalize = Normalize(self.IMAGE2_MEAN, self.IMAGE2_STD)
        self.depth_divide = Normalize(0, self.DEPTH_NORMALIZER)


    def transform_img0(self, img):
        img = self.make_tensor(img)
        return self.img0_normalize(img).double()

    def transform_img1(self, img):
        img = self.make_tensor(img)
        return self.img1_normalize(img).double()

    def transform_img2(self, img):
        img = self.make_tensor(img)
        return self.img2_normalize(img).double()
    
    def transform_depths(self, depths):
        
        depths = self.make_tensor(depths).swapaxes(0,1)
        return self.depth_divide(depths).double()
    

    def __len__(self):
        return len(self.data_folders)


    def __getitem__(self, idx):
        f = self.data_folders[idx]

        img0 = self.transform_img0(Image.open(os.path.join(self.pathX, f, 'rgb', '0.png')))
        img1 = self.transform_img1(Image.open(os.path.join(self.pathX, f, 'rgb', '1.png')))
        img2 = self.transform_img2(Image.open(os.path.join(self.pathX, f, 'rgb', '2.png')))

        depth = self.transform_depths(np.load(os.path.join(self.pathX, f, 'depth.npy')))

        field_id = pkl.load(open(os.path.join(self.pathX, f, 'field_id.pkl'), 'rb'))

        finger_positions = np.load(os.path.join(self.pathY, f + '.npy'))

        return (img0, img1, img2, depth, field_id), finger_positions
