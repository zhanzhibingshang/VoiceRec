from torch.utils.data import DataLoader, Dataset
import os
import torch
import glob
import numpy as np
import random
import h5py

class MicroPhoneDataset(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        #self.transform = transform  # 变换
        self.images = sorted(glob.glob(os.path.join(self.root_dir,'*.npy')))
        self.images = self.images

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)
        #return len(64)

    def get_heatmap(self,filepath):
        x = []
        y = []
        rms = []
        begin = 3
        count_x = 0
        count_y = 0
        map_w = 32
        map_h = 32
        if 'x4' in filepath:
            time_count = 4
        elif 'x3' in filepath:
            time_count = 3
        elif 'x2' in filepath:
            time_count = 2
        elif 'x' in filepath:
            time_count = 1

        for count in range(time_count):
            x.append(float(filepath.split('_')[begin + 6 * count_x]))
            count_y += 1
            y.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_y += 1
            rms.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_x += 1
            count_y = 0

        heatmap_all = np.zeros((map_h, map_w), dtype=np.uint8)
        for count in range(time_count):
            x_input = x[count]
            y_input = y[count]
            rms_input = rms[count]

            x_input = int(x_input * map_w / 3.) + map_h // 2 - 1
            y_input = int(y_input * map_w / 3.) + map_h // 2 - 1
            heatmap_all[x_input,y_input] = 1

        return heatmap_all

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        #image_index = self.images[index]  # 根据索引index获取该图片
        #img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        TimeData = np.load(self.images[index])
        #TimeData = np.dot(TimeData,TimeData.transpose())

        # label_name = os.path.join(self.root_dir,'Location_np',self.images[index].split('/')[-1])
        # label = np.load(label_name)


        #clip = random.randint(0,50000)
        #print(TimeData.shape,clip,self.images[index])
        #TimeData = TimeData[clip:clip+1024,:]
        label = self.get_heatmap(self.images[index])
        TimeData = torch.Tensor(TimeData).unsqueeze(0)
        #img = torch.rand(1024,64,1)
        label = torch.Tensor(label).long()
        #label = torch.rand(1,32,32)
        #print(label.max())

        return TimeData,label

class MicroPhoneDatasetRNN(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        #self.transform = transform  # 变换
        self.images = sorted(glob.glob(os.path.join(self.root_dir,'VoiceData','*.npy')))
        self.images = self.images

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)
        #return len(64)

    def get_heatmap(self,filepath):
        x = []
        y = []
        rms = []
        begin = 2
        count_x = 0
        count_y = 0
        map_w = 32
        map_h = 32
        if 'x4' in filepath:
            time_count = 4
        elif 'x3' in filepath:
            time_count = 3
        elif 'x2' in filepath:
            time_count = 2
        elif 'x' in filepath:
            time_count = 1

        for count in range(time_count):
            x.append(float(filepath.split('_')[begin + 6 * count_x]))
            count_y += 1
            y.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_y += 1
            rms.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_x += 1
            count_y = 0

        heatmap_all = np.zeros((map_h, map_w), dtype=np.uint8)
        for count in range(time_count):
            x_input = x[count]
            y_input = y[count]
            rms_input = rms[count]

            x_input = int(x_input * map_w / 3.) + map_h // 2 - 1
            y_input = int(y_input * map_w / 3.) + map_h // 2 - 1
            heatmap_all[x_input,y_input] = 1

        return heatmap_all

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        #image_index = self.images[index]  # 根据索引index获取该图片
        #img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        TimeData = np.load(self.images[index])
        TimeData = TimeData.transpose()
        #TimeData = np.dot(TimeData,TimeData.transpose())

        # label_name = os.path.join(self.root_dir,'Location_np',self.images[index].split('/')[-1])
        # label = np.load(label_name)


        #clip = random.randint(0,50000)
        #print(TimeData.shape,clip,self.images[index])
        #TimeData = TimeData[clip:clip+1024,:]
        label = self.get_heatmap(self.images[index])
        TimeData = torch.Tensor(TimeData)
        #img = torch.rand(1024,64,1)
        label = torch.Tensor(label).long()
        #label = torch.rand(1,32,32)
        #print(label.max())

        return TimeData,label

class MicroPhoneDatasetAu(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        # self.transform = transform  # 变换
        self.images1 = sorted(glob.glob(os.path.join(self.root_dir,'one_source' '/*.h5')))
        self.images2 = sorted(glob.glob(os.path.join(self.root_dir,'two_sources' '/*.h5')))
        self.images3 = sorted(glob.glob(os.path.join(self.root_dir,'three_sources' '/*.h5')))
        self.images4 = sorted(glob.glob(os.path.join(self.root_dir,'four_sources' '/*.h5')))
        self.images = self.images1 + self.images2  +  self.images3  +  self.images4

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)
        # return len(64)

    def get_heatmap(self, filepath):
        x = []
        y = []
        rms = []
        begin = 3
        count_x = 0
        count_y = 0
        map_w = 32
        map_h = 32
        if 'x4' in filepath:
            time_count = 4
        elif 'x3' in filepath:
            time_count = 3
        elif 'x2' in filepath:
            time_count = 2
        elif 'x' in filepath:
            time_count = 1

        for count in range(time_count):
            x.append(float(filepath.split('_')[begin + 6 * count_x]))
            count_y += 1
            y.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_y += 1
            rms.append(float(filepath.split("_")[begin + count_y * 2 + 6 * count_x]))
            count_x += 1
            count_y = 0

        heatmap_all = np.zeros((map_h, map_w), dtype=np.uint8)
        for count in range(time_count):
            x_input = x[count]
            y_input = y[count]
            rms_input = rms[count]

            x_input = int(x_input * map_w / 3)  + map_h // 2 - 1
            y_input = int(y_input * map_w / 3.) + map_h // 2 - 1
            heatmap_all[x_input, y_input] = 1

        return heatmap_all

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # image_index = self.images[index]  # 根据索引index获取该图片
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        TimeData = h5py.File(self.images[index],'r')
        TimeData = TimeData['time_data'].value
        # TimeData = np.dot(TimeData,TimeData.transpose())

        # label_name = os.path.join(self.root_dir,'Location_np',self.images[index].split('/')[-1])
        # label = np.load(label_name)

        clip = random.randint(0,51200-1025)
        #print(TimeData.shape,clip,self.images[index])
        TimeData = TimeData[clip:clip+1024,:]
        label = self.get_heatmap(self.images[index])
        TimeData = torch.Tensor(TimeData).unsqueeze(0)
        # img = torch.rand(1024,64,1)
        label = torch.Tensor(label).long()
        # label = torch.rand(1,32,32)
        # print(label.max())

        return TimeData, label