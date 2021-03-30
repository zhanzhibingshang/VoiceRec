from torch.nn import Module
import torch.nn as nn
import torch

class VoiceRecNet(Module):
    def __init__(self):
        super(VoiceRecNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1),
                                    nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv10 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,stride=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = nn.functional.interpolate(x,size=(64,1))
        x = x.permute(0,3,1,2)
        x = self.conv6(x)
        x = self.conv7(x)
        x= self.conv8(x)
        x = self.conv9(x)
        x= self.conv10(x)
        x = self.conv11(x)
        return x

class VoiceRecNet2(Module):
    def __init__(self):
        super(VoiceRecNet2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=2,padding=1),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=2,padding=1),
                                    nn.ReLU())
        self.conv4= nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=2,padding=1),
                                    nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3,stride=1,padding=1),
                                   nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1,stride=1)


    def forward(self, x):
        #print(x.shape)
        weight1 = torch.nn.Parameter(torch.rand(x.shape),requires_grad=True).cuda()
        weight2 = torch.nn.Parameter(torch.rand(x.shape),requires_grad=True).cuda()
        x1 = x * weight1
        x2 = x * weight2
        #print(x1.squeeze(-1).shape,x2.shape)
        #x1 = x
        #x2 = x
        x = torch.bmm(x1.squeeze(1),x2.squeeze(1).permute(0,2,1))
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        #x = torch.nn.functional.sigmoid(x#)
        return x


class VoiceRecNet3(Module):
    def __init__(self):
        super(VoiceRecNet3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3),stride=(2,1),padding=(1,1)),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3),stride=(2,1),padding=(1,1)),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3),stride=(2,1),padding=(1,1)),
                                    nn.ReLU())
        self.conv4= nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),stride=(2,1),padding=(1,1)),
                                    nn.ReLU())
        self.conv5= nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                    nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,stride=1,padding=1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1,stride=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
       # print(x.shape)
        x = torch.nn.functional.interpolate(x,(32,32))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        #print(x.shape)
        #x = torch.nn.functional.sigmoid(x#)
        return x

class VoiceRecRNN(Module):
    def __init__(self):
        super(VoiceRecRNN, self).__init__()
        self.rnn1 = nn.LSTM(input_size=1024,
                           hidden_size=64,
                           num_layers=1,
                           batch_first=True)
        self.fc1 = nn.Linear(64,1024)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,stride=1,padding=1),
                                    nn.ReLU())
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1,stride=1)


    def forward(self, x):
        batch = x.shape[0]
        x,(h_n,h_c) = self.rnn1(x,None)

        x = self.fc1(x[:,-1,:])
        x = x.view(batch,1,32,32)
        x = self.conv1(x)
        x = self.conv2(x)
        return x