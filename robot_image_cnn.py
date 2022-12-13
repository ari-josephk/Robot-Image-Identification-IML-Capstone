import torch
import torch.nn as nn
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class Robot_Image_CNN(nn.Module):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, use_img0 = True, use_img1 = True, use_img2=True, conv_feature = 56, fc_feature = 28, output_size = 12):
        super(Robot_Image_CNN, self).__init__()
        input_channels = (4 if use_img0 else 0) + (4 if use_img1 else 0) + (4 if use_img2 else 0)
        fcn_input_size = 157304

        self.use_img0 = use_img0
        self.use_img1 = use_img1
        self.use_img2 = use_img2

        #define our functions
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LayerNorm((1, 12))
        self.conv2D1 = nn.Conv2d(input_channels, conv_feature, kernel_size= 5)
        self.conv2D2 = nn.Conv2d(conv_feature, conv_feature, kernel_size= 5)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(fcn_input_size, fc_feature)
        self.linear2 = nn.Linear(fc_feature, output_size)

        self.double()

        self.optimizer = torch.optim.SGD(self.parameters(), 0.01, 0.5)
        self.get_loss = RMSELoss()

        self.to(self.DEVICE)

    def forward(self, x):
        #create the architecture for the model
        x = self.sigmoid(self.conv2D1(x))
        x = self.maxpool2D(x)
        x = self.sigmoid(self.conv2D2(x))
        x = self.maxpool2D(x) 
        x = self.flatten(x)
        x = self.sigmoid(self.linear1(x))
        return self.linear2(x)

    def train(self, train_loader, epoch = 0):
        super(Robot_Image_CNN, self).train()
        
        num_elements = len(train_loader.dataset)
        batch_size = train_loader.batch_size

        for batch_idx, (data, target) in enumerate(train_loader):
            #if batch_idx == 2:
            #    break
            # send to device
            img0, img1, img2, depths, field_id = data
            
            img0 = torch.concat((img0, depths[:,0:1]), 1) if self.use_img0 else torch.Tensor()
            img1 = torch.concat((img1, depths[:,1:2]), 1) if self.use_img1 else torch.Tensor()
            img2 = torch.concat((img2, depths[:,2:3]), 1) if self.use_img2 else torch.Tensor()

            data = torch.concat((img0, img1, img2), 1)

            data, target = data.to(self.DEVICE), target.to(self.DEVICE)

            self.optimizer.zero_grad()
            output = self(data)

            loss = self.get_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 1 == 0:
                print('Train Epoch {}, Batch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, batch_idx * train_loader.batch_size, num_elements,
                    100. * (batch_idx * batch_size) / num_elements, loss.item()))

    def test(self, test_loader, epoch = 0, return_field_ids=False):
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
                print('Test Epoch {}, Batch {}: [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx, batch_idx * test_loader.batch_size, num_elements,
                    100. * (batch_idx * batch_size) / num_elements))
        
        return field_ids, preds if return_field_ids else preds

            