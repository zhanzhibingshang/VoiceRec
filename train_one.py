import torch
import argparse
from dataset import MicroPhoneDataset
from torch.utils.data import DataLoader
from model import *
import os
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description='Reconstruct the voice')
    parser.add_argument('--train_bs',type=int, default=32,help='batchsize of train')
    parser.add_argument('--val_bs',type=int, default=32,help='batchsize of val')
    parser.add_argument('--num_worker',type=int, default=16,help='batchsize of train')
    parser.add_argument('--lr', type=float,default=0.001,help='learningRate of train')
    parser.add_argument('--epoch', type=int,default=1000,help='epoch of train')
    args = parser.parse_args()
    return args

def main(args):
    ## defined the Dataset and DataLoader
    train_dataset = MicroPhoneDataset(root_dir='/home3/zengwh/VoiceRec/data/compute/100000_data/Train/one_source')
    val_dataset = MicroPhoneDataset(root_dir='/home3/zengwh/VoiceRec/data/compute/100000_data/Val/one_source')

    train_dataloader = DataLoader(train_dataset,batch_size=args.train_bs,shuffle=True,num_workers=args.num_worker,drop_last=False)
    val_dataloader = DataLoader(val_dataset,batch_size=args.val_bs,shuffle=False,num_workers=args.num_worker,drop_last=False)

    ## defined the network and Loss
    Model = VoiceRecNet3()
    if torch.cuda.is_available():
        Model = Model.cuda()
        #Model = torch.nn.DataParallel(Model)
    #print(summary(VoiceRecNet,(1,1024,56,1)))
    #loss = torch.nn.MSELoss()
    loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.1,0.9]).cuda())

    ## defined the optimizer
    # optim = torch.optim.SGD(Model.parameters(),lr=args.lr,
    #             momentum=0.9,
    #             dampening=0,
    #             weight_decay=0,
    #             nesterov=False)
    ## defined the optimizer
    optim = torch.optim.Adam(Model.parameters(),lr=args.lr,
                             betas=(0.9,0.999),
                             eps=1e-8,
                             weight_decay=1e-5,
                             amsgrad=False)

    lr_sche = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[500,700,900], gamma=0.1, last_epoch=-1)

    ## define tensorboard
    write = SummaryWriter(comment='first', log_dir='./log/',filename_suffix='first')
    #dummy_input = torch.rand(1,1024,56,1)
    #write.add_graph(VoiceRecNet,input_to_model = dummy_input)

    ## train and val
    for epoch in range(args.epoch):
        ## train stage
        total_train_loss = 0.
        Model.train()
        for batch_idx,(img,label) in enumerate(train_dataloader):

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()


            pre = Model(img)
            #pre = torch.clamp(pre,0.,1.)
            batch_loss = loss(pre,label) + 0.1*torch.abs(pre).mean()

            optim.zero_grad()
            batch_loss.backward()
            optim.step()
            total_train_loss += batch_loss.item()

            if batch_idx == 0:
                write.add_images('train_gt',label.unsqueeze(1),global_step=epoch)
                temp = torch.nn.functional.sigmoid(pre[:,1,:,:].cpu()).unsqueeze(1)>0.5
                write.add_images("train_pre",temp.float(),global_step=epoch)


            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tbatch loss: {:.6f}\tAvg loss: {:.6f}'.
                    format(
                    epoch,
                    (batch_idx+1) * len(img),
                    len(train_dataset),
                    100. * (batch_idx+1) / len(train_dataloader),
                    batch_loss.item(),total_train_loss/(batch_idx+1)))
        lr_sche.step()
        write.add_scalar('train_loss',total_train_loss/(batch_idx+1),global_step=epoch)
        write.add_scalar('lr',optim.state_dict()['param_groups'][0]["lr"],global_step=epoch)

        print("begin val")
        ## val stage
        Model.eval()
        total_val_loss = 0.
        for batch_idx,(img,label) in enumerate(val_dataloader):
            img = img.cuda()
            label = label.cuda()


            pre = Model(img)
            #pre = torch.clamp(pre,0.,1.)
            batch_loss = loss(pre,label)

            total_val_loss += batch_loss.item()

            if batch_idx == 0:
                write.add_images('val_gt',label.unsqueeze(1),global_step=epoch)
                temp = torch.nn.functional.sigmoid(pre[:,1,:,:].cpu()).unsqueeze(1)>0.5
                write.add_images("val_pre",temp.float(),global_step=epoch)


        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tbatch loss: {:.6f}\tAvg loss: {:.6f}'.
            format(
            epoch,
            (batch_idx+1) * len(img),
            len(val_dataset),
            100. * (batch_idx+1) / len(train_dataloader),
            batch_loss.item(),total_val_loss/(batch_idx+1)))

        write.add_scalar('val_loss',total_val_loss/(batch_idx+1),global_step=epoch)
        os.makedirs('models',exist_ok=True)
        torch.save(Model.state_dict(),'./models/model.pth')
if __name__ == "__main__":
    args = get_args()
    main(args)