import torch
import argparse
from dataset import MicroPhoneDataset
from torch.utils.data import DataLoader
from model import *
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
# from torchsummary import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_args():
    parser = argparse.ArgumentParser(description='Reconstruct the voice')
    parser.add_argument('--train_bs', type=int, default=32, help='batchsize of train')
    parser.add_argument('--val_bs', type=int, default=1, help='batchsize of val')
    parser.add_argument('--num_worker', type=int, default=16, help='batchsize of train')
    parser.add_argument('--lr', type=float, default=0.001, help='learningRate of train')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch of train')
    args = parser.parse_args()
    return args


def main(args):
    ## defined the Dataset and DataLoader
    val_dataset = MicroPhoneDataset(root_dir='/home3/zengwh/VoiceRec/data/compute/100000_data/Val/one_source')
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, num_workers=args.num_worker,
                                drop_last=False)
    ## defined the network and Loss
    Model = VoiceRecNet3()
    if torch.cuda.is_available():
        Model = Model.cuda()



    print("begin val")
    ## val stage
    Model.eval()
    total_val_loss = 0.
    avg_acc, avg_pre, avg_recall, avg_f1 = 0., 0., 0., 0.
    for batch_idx, (img, label) in enumerate(val_dataloader):
        img = img.cuda()
        label = label
        label = torch.flatten(label).cpu().numpy()

        pre = Model(img)
        #pre = torch.nn.functional.sigmoid(pre[:, 1, :, :].cpu()).unsqueeze(1)
        pre = pre[:, 1, :, :]
        # print(pre)
        #pre = pre > 0.5
        pre = torch.flatten(pre.long()).cpu().numpy()
        print(max(pre))
        print(np.where(pre==np.max(pre)))
        print('pre max',np.max(pre))
        # np.where(]
        # pre[np.where(pre==np.max(pre))] = 1
        # pre[np.where(pre==np.max(pre))] = 0

        # print(pre)
        # print('pre max',np.max(pre))
        print('num of pre',sum(pre))
        print('num of label',sum(label))
        # print('temp',temp.shape)
        # print('report',classification_report(label.cpu().numpy(),temp.numpy()))
        avg_acc += accuracy_score(label,pre)
        avg_pre += precision_score(label,pre)
        avg_recall += recall_score(label,pre)
        avg_f1 += f1_score(label,pre)
    print('average accuracy:',avg_acc/len(val_dataset))
    print('average precision:',avg_pre/len(val_dataset))
    print('average recall:',avg_recall/len(val_dataset))
    print('average f1:',avg_f1/len(val_dataset))




if __name__ == "__main__":
    args = get_args()
    main(args)