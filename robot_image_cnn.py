import torch
import torch.nn as nn
import numpy as np

'''
MinPool2d
Simple implementation of minpooling, using pytorch's maxpool. 
Because the images are a dark subject on a light background, I expected this 
to work slightly better than Maxpool

Params
    kernel_size - the size of the window to take a min over
'''
class MinPool2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=2)

    def forward(self, x):
        return self.maxpool(-1 * x).detach() * -1


'''
Robot_Image_CNN
Implementation of a CNN to identify finger positions from robot hand images and depth matrices
Implmented a CNN that is similar to the VGG16 architecture, but a lot computationally lighter so that it's able to run on regular computers quickly. 
It includes 10 Conv layers with different versions of ReLU nonlinearity, 4 pooling layers, and then a FCN with 3 linear layers with regular ReLu,
and uses the Adam algorithm for parameter optimization. 

Constants
    DEVICE - automatically selected device to run the model on. You should probably have a strong GPU or you will have a bad time.
    POOL_KERNEL_SIZE - kernel size parameter for pooling layers
    CONV1_CHANNELS - Channels for first group of Conv layers
    CONV2_CHANNELS - Channels for second group of Conv layers
    CONV2_CHANNELS - Channels for third group of Conv layers
    CONV2_CHANNELS - Channels for last group of Conv layers
    LEARNING_RATE  - Leaning rate for Adam algoritm
    FC_FEATURE - Features for the linear layers of the FCN
    CONV_KERNEL_SIZE - kernel size parameter for Conv layers

Params:
    use_img0-2 - True if we want to use img0-2 in the model, else False
    output_size - size of the output vector, 12 for this problem by default

'''
class Robot_Image_CNN(nn.Module):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    POOL_KERNEL_SIZE = 3
    CONV1_CHANNELS = 64
    CONV2_CHANNELS = 128
    CONV3_CHANNELS = 256
    CONV4_CHANNELS = 512
    LEARNING_RATE = 1e-4
    FC_FEATURE = 4096
    CONV_KERNEL_SIZE = 3

    def __init__(self, use_img0 = True, use_img1 = True, use_img2=True, output_size = 12):
        super(Robot_Image_CNN, self).__init__()
        input_channels = (4 if use_img0 else 0) + (4 if use_img1 else 0) + (4 if use_img2 else 0) #4 channels for each image (RGB + depth)
        fcn_input_size = 86528 #We could compute this automatically if needed

        self.use_img0 = use_img0
        self.use_img1 = use_img1
        self.use_img2 = use_img2

        #define our functions
        self.flatten = nn.Flatten()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.rrelu = nn.RReLU()

        #define our convolution and pooling layers
        self.conv2D1_1 = nn.Conv2d(in_channels=input_channels, out_channels=self.CONV1_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D1_2 = nn.Conv2d(in_channels=self.CONV1_CHANNELS, out_channels=self.CONV1_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D2_1 = nn.Conv2d(in_channels=self.CONV1_CHANNELS, out_channels=self.CONV2_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D2_2 = nn.Conv2d(in_channels=self.CONV2_CHANNELS, out_channels=self.CONV2_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D3_1 = nn.Conv2d(in_channels=self.CONV2_CHANNELS, out_channels=self.CONV3_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D3_2 = nn.Conv2d(in_channels=self.CONV3_CHANNELS, out_channels=self.CONV3_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D3_3 = nn.Conv2d(in_channels=self.CONV3_CHANNELS, out_channels=self.CONV3_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D4_1 = nn.Conv2d(in_channels=self.CONV3_CHANNELS, out_channels=self.CONV4_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D4_2 = nn.Conv2d(in_channels=self.CONV4_CHANNELS, out_channels=self.CONV4_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.conv2D4_3 = nn.Conv2d(in_channels=self.CONV4_CHANNELS, out_channels=self.CONV4_CHANNELS, kernel_size=self.CONV_KERNEL_SIZE, padding=1)
        self.maxpool2D = MinPool2d(kernel_size=self.POOL_KERNEL_SIZE)

        #define our linear layers
        self.linear1 = nn.Linear(fcn_input_size, self.FC_FEATURE)
        self.linear2 = nn.Linear(self.FC_FEATURE, self.FC_FEATURE)
        self.linear3 = nn.Linear(self.FC_FEATURE, output_size)

        #define our optimizer and loss measure
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        self.get_loss = nn.L1Loss() #we use L1Loss over RMSE/MSE because its much faster, for similar result in the long run

        self.to(self.DEVICE) #send model to GPU if needed

    def forward(self, x):
        #first group of conv and pooling
        x = self.rrelu(self.conv2D1_1(x))
        x = self.rrelu(self.conv2D1_2(x))
        x = self.maxpool2D(x)
        
        #second group of conv and pooling
        x = self.rrelu(self.conv2D2_1(x))
        x = self.rrelu(self.conv2D2_2(x))
        x = self.maxpool2D(x)

        #third group of conv and pooling, using leaky_relu instead of random now
        x = self.leaky_relu(self.conv2D3_1(x))
        x = self.leaky_relu(self.conv2D3_2(x))
        x = self.leaky_relu(self.conv2D3_3(x))
        x = self.maxpool2D(x)

        #last group of conv and pooling,
        x = self.leaky_relu(self.conv2D4_1(x))
        x = self.leaky_relu(self.conv2D4_2(x))
        x = self.leaky_relu(self.conv2D4_3(x))
        x = self.maxpool2D(x)

        #flatten and apply our linear layers with FCN
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


    '''
    Train the model
    Params:
        train_loader - dataloader with the train data
        num_epoch - number of epoches to run on the train data, default 1
    '''
    def train(self, train_loader, num_epoch = 1):
        super(Robot_Image_CNN, self).train()
        
        num_elements = len(train_loader.dataset)
        batch_size = train_loader.batch_size

        for epoch in range(num_epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                #format the data into RGB + Depth channels for each image we are using, and send to device
                img0, img1, img2, depths, field_id = data
                
                img0 = torch.concat((img0, depths[:,0:1]), 1) if self.use_img0 else torch.Tensor()
                img1 = torch.concat((img1, depths[:,1:2]), 1) if self.use_img1 else torch.Tensor()
                img2 = torch.concat((img2, depths[:,2:3]), 1) if self.use_img2 else torch.Tensor()

                data = torch.concat((img0, img1, img2), 1)

                data, target = data.to(self.DEVICE), target.to(self.DEVICE)

                #use optimizers to adjust parameters, and keep track of the loss at each batch
                self.optimizer.zero_grad()
                output = self(data)

                loss = self.get_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 1 == 0:
                    print('Train Epoch {}, Batch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx, batch_idx * train_loader.batch_size, num_elements,
                        100. * (batch_idx * batch_size) / num_elements, loss.item()))


    '''
    Get predictions for our test set
    Params
        test_loader - dataloader with the rest data
        return_field_ids - True if we want to return the field ids of the passed in data, else False

    Returns:
        field_ids - numpy array of the passed in test set, in order (ONLY returned if return_field_ids=True)
        preds - Tensor of model predictions for the test set
    '''
    def test(self, test_loader, return_field_ids=False):
        super(Robot_Image_CNN, self).train(False)

        num_elements = len(test_loader.dataset)
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size
        preds = torch.zeros([num_elements, 12])

        field_ids = np.array([])

        for batch_idx, (data, target) in enumerate(test_loader):
            # send to device
            img0, img1, img2, depths, field_id = data
            
            img0 = torch.concat((img0, depths[:,0:1]), 1) if self.use_img0 else torch.Tensor()
            img1 = torch.concat((img1, depths[:,1:2]), 1) if self.use_img1 else torch.Tensor() 
            img2 = torch.concat((img2, depths[:,2:3]), 1) if self.use_img2 else torch.Tensor()

            data = torch.concat((img0, img1, img2), 1)

            data = data.to(self.DEVICE)

            start = batch_idx*batch_size
            end = start + batch_size
            if batch_idx == num_batches - 1:
                end = num_elements
             
            pred = self(data)
            #print(batch_preds)

            preds[start:end] = pred.clone().detach()
            field_ids = np.append(field_ids, field_id)

            if batch_idx % 1 == 0:
                print('Testing Set, Batch {}: [{}/{} ({:.0f}%)]'.format(
                    batch_idx, batch_idx * test_loader.batch_size, num_elements,
                    100. * (batch_idx * batch_size) / num_elements))
        
        return field_ids, preds if return_field_ids else preds

            