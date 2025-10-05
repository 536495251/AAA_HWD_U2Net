from datetime import datetime
import numpy as np
import torch
import time
import os
import webbrowser
import subprocess
import os.path as ops
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets.MyDataset import MyDataset
from models.adan import Adan
from models.loss.loss import criterion, SoftIoULoss
from models.lr_scheduler import create_lr_scheduler, adjust_learning_rate
from models.metrics import IoUMetric, nIoUMetric, PD_FA
from utils.get_config import get_config
from utils.model_creater import model_creater


class Train:
    def __init__(self):
        self.train = get_config('train')
        self.net_config = get_config('net_config')
        self.dataset=self.train['dataset']
        self.train_dataset=MyDataset(self.dataset,'train')
        self.val_dataset=MyDataset(self.dataset,'val')
        self.image_size =self.train_dataset.data.image_size
        self.train_dataloader=DataLoader(dataset=self.train_dataset,batch_size=self.train['batch_size'],shuffle=self.train['shuffle'],
                                         num_workers=self.train['num_workers'],pin_memory=self.train['pin_memory'])
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.train['batch_size'],shuffle=self.train['shuffle'],
                                           num_workers=self.train['num_workers'], pin_memory=self.train['pin_memory'])
        self.net=model_creater(self.train).get_net().cuda()
        self.model_name=model_creater(self.train).get_model_name()
        self.scaler = torch.amp.GradScaler('cuda') if self.train['amp'] else None
        self.criterion = criterion
        self.optimizer = Adan(self.net.parameters(), lr=self.train['learning_rate'], weight_decay=float(self.train['weight_decay']))

        self.lr_scheduler = create_lr_scheduler(self.optimizer, len(self.train_dataloader), self.train['epochs'], warmup=self.train['warmup'],
                                                warmup_epochs=self.train['warm_up_epochs'])
        if self.train['resume'] != '':
            checkpoint = torch.load(self.train['resume'])
            print(checkpoint['net'].keys())
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.train['start_epoch'] = checkpoint['epoch'] + 1
            if self.train['amp']:
                self.scaler.load_state_dict(checkpoint["scaler"])
        # 初始化评估指标和状态
        self.iou_metric = IoUMetric()
        self.nIoU_metric = nIoUMetric(1, score_thresh=0.5)
        self.best_iou = 0
        self.best_nIoU = 0
        self.best_PD = 0
        self.best_FA = 1
        self.PD_FA = PD_FA(self.image_size)

        if self.train['resume'] != '':
            self.folder_name = os.path.abspath(
                os.path.dirname(os.path.abspath(os.path.dirname(self.train['resume']) + os.path.sep + "."))
                + os.path.sep + ".")
        else:
            self.folder_name = '%s_bs%s_lr%s' % (time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())),
                                            self.train['batch_size'], self.train['learning_rate'])

        time_str = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        self.save_name =str(self.train['dataset'])
        self.save_folder = ops.join(self.model_name+'_At_' + self.save_name, self.folder_name)
        self.save_log = ops.join(self.save_folder, 'log')
        self.save_pth = ops.join(self.save_folder, 'checkpoint')
        os.makedirs(self.save_pth, exist_ok=True)
        os.makedirs(self.save_log, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_log)

        self.train['tensorboard_logdir'] = self.save_log

    def training(self, epoch):
        losses = []
        self.net.train()
        tbar = tqdm(self.train_dataloader)
        for i, (data, labels) in enumerate(tbar):
            data, labels = data.cuda(), labels.cuda()
            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                output = self.net(data)
                loss = self.criterion(output, labels)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            losses.append(loss.item())
            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f'% (epoch, self.optimizer.param_groups[0]['lr'], np.mean(losses)))
            # adjust_learning_rate(self.optimizer, epoch, self.train['epochs'], self.train['learning_rate'],
            #                   self.train['warm_up_epochs'], 1e-6)
        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):

        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        eval_losses = []
        self.net.eval()
        tbar = tqdm(self.val_dataloader)
        for i, (data, labels) in enumerate(tbar):
            with torch.no_grad():
                output = self.net(data.cuda())
                output = output.cpu()

            loss = self.criterion(output, labels)
            eval_losses.append(loss.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.PD_FA.update(output, labels)

            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            Fa, Pd = self.PD_FA.get(len(self.val_dataset))

            tbar.set_description('  Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f, Fa:%.8f, Pd:%.8f'
                                 % (epoch, np.mean(eval_losses), IoU, nIoU, Fa, Pd))

        pkl_name = 'Epoch=%3d_IoU=%.4f_nIoU=%.4f_Fa=%.8f_Pd=%.8f.pth' % (epoch, IoU, nIoU, Fa, Pd)
        save_file = {"net": self.net.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "lr_scheduler": self.lr_scheduler.state_dict(),
                     "epoch": epoch,
                    }
        if self.train['amp']:
            save_file["scaler"] = self.scaler.state_dict()
        save_pth = self.save_pth
        if IoU > self.best_iou:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_nIoU = nIoU
        if Pd > self.best_PD:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_PD = Pd
        if Fa < self.best_FA:
            torch.save(save_file, ops.join(save_pth, pkl_name))
            self.best_FA = Fa

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        self.writer.add_scalar('Eval/Pd', Pd, epoch)
        self.writer.add_scalar('Eval/Fa', Fa, epoch)
        self.writer.add_scalar('Best/Pd', self.best_PD, epoch)
        self.writer.add_scalar('Best/Fa', self.best_FA, epoch)
if __name__ == '__main__':
    train = Train()
    tensorboard_logdir=train.train['tensorboard_logdir']
    print(tensorboard_logdir)
    for epoch in range(train.train['start_epoch'],  train.train['epochs']+ 1):
        train.training(epoch)
        train.validation(epoch)
    print('Best IoU: %.5f, best nIoU: %.5f, Best Pd: %.5f, best Fa: %.5f' %(train.best_iou, train.best_nIoU, train.best_PD, train.best_FA))