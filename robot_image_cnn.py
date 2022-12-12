import torch
import torch.nn as nn

class Robot_Image_CNN(nn.Module):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_channels=4, conv_feature=56, fc_feature=28, output_size=12):
        super.__init__()
        self.optimizer = torch.optim.SGD(self.parameters(), 0.01, 0.5)
        self.get_loss = nn.CrossEntropyLoss()

        #define our functions
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.conv2D1 = nn.Conv2d(input_channels, conv_feature, kernel_size= 5)
        self.conv2D2 = nn.Conv2d(conv_feature, conv_feature, kernel_size= 5)
        self.maxpool2D = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(conv_feature * 4 * 4, fc_feature)
        self.linear2 = nn.Linear(fc_feature, output_size)
        
    def forward(self, x):
        #create the architecture for the model
        x = self.relu(self.conv2D1(x))
        x = self.maxpool2D(x)
        x = self.relu(self.conv2D2(x))
        x = self.maxpool2D(x) 
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        return self.log_softmax(self.linear2(x))

    def train(self, epoch, train_loader):
        super.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # send to device
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)

            self.optimizer.zero_grad()
            output = self(data)

            loss = self.get_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))