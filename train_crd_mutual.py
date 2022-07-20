# coding=utf-8


import gc
import sys
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from conf import model_config_bert as model_config
import os
import pandas as pd
import torch
import numpy as np
from helper import *
from crd.criterion import CRDLoss
from torch.utils.data import Dataset
from transformers import BertTokenizer
from model.resnet import resnet50
from model.bert import Model
from conf import config
from PIL import Image
from KD import DistillKL
import torchvision.transforms as transforms
from PIL import ImageFile

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(Dataset):

    def __init__(self, df, mode='train', task='2', k=1586):
        self.mode = mode
        self.task = task
        self.k = k
        self.tokenizer = BertTokenizer.from_pretrained(model_config.pretrain_model_path)
        self.pad_idx = self.tokenizer.pad_token_id
        self.id = []
        self.x_data = []
        self.img_data = []
        self.y_data = []
        self.cls_positive = [[] for _ in range(3)]
        self.cls_negative = [[] for _ in range(3)]

        for i, row in df.iterrows():
            x, y = self.row_to_tensor(self.tokenizer, row)
            img_name = row['img']
            # img_path = '/home/aistudio/train_images/' + img_name
            self.id.append(row['id'])
            self.x_data.append(x)
            self.img_data.append(self.read_img(img_name))
            self.y_data.append(y)
            self.cls_positive[y.numpy()].append(i)
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(3)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(3)]
        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        pos_idx = index
        replace = True if self.k > len(self.cls_negative[self.y_data[index].numpy()]) else False
        neg_idx = np.random.choice(self.cls_negative[self.y_data[index].numpy()], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return self.id[index], self.x_data[index], self.img_data[index], self.y_data[index], index, sample_idx

    def contact(self, str1, str2):
        if pd.isnull(str2):
            return str1
        return str1 + str2

    def read_img(self,img_name):
        img_path='/home/aistudio/train_images/'+img_name
        img = Image.open(img_path).convert('RGB')
        transform_train_list = [
            # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        transform_test_list = [
            transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        tran_trains = transforms.Compose(transform_train_list)
        tran_tests = transforms.Compose(transform_test_list)
        if self.mode == 'train':
            img = tran_trains(img)
        if self.mode=='val':
            img = tran_tests(img)
        res_img = img.float()
        return res_img

    def row_to_tensor(self, tokenizer, row):
        if self.task == '0':
            text = row['content']
        elif self.task == '1':
            text = self.contact(row['content'], row['comment_2c'])
        else:
            text = self.contact(row['content'], row['comment_all'])
        x_encode = tokenizer.encode(text)
        if len(x_encode) > config.max_seq_len[self.task]:
            text_len = int(config.max_seq_len[self.task] / 2)
            x_encode = x_encode[:text_len] + x_encode[-text_len:]
        else:
            padding = [0] * (config.max_seq_len[self.task] - len(x_encode))
            x_encode += padding
        x_tensor = torch.tensor(x_encode, dtype=torch.long)
        if self.mode == 'test':
            y_tensor = torch.tensor([0] * len(config.label_columns), dtype=torch.long)
        else:
            y_data = row['label']
            y_tensor = torch.tensor(y_data, dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.y_data)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=30, type=int, help="epochs num")
    parser.add_argument("-m", "--model_name", default='resnet', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=2, type=int, help="train mode")
    parser.add_argument("-re", "--resume", type=str, default='data/model/1/text_bert_task2_trial1.bin', metavar='PATH')
    parser.add_argument("-task", "--task", default='2', type=str, help="task")
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--distill', type=str, default='nst', choices=['kd', 'nst'])
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.8, help='weight balance for other losses')

    parser.add_argument('--nce_k', default=1586, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    args = parser.parse_args()
    args.save_folder = os.path.join(config.model_path, 'crd')
    return args


def get_label(row):
    if row['ncw_label'] == 1:
        return 0
    if row['fake_label'] == 1:
        return 1
    if row['real_label'] == 1:
        return 2


def get_img(row):
    img = row['picture_lists'].split('\t')[0]
    return img


def img_exist(img):
    if os.path.exists('/home/aistudio/train_images/' + img):
        return True
    return False


def main():
    best_img_acc = 0
    best_text_acc = 0

    args = parse_option()

    sys.stdout = Logger(os.path.join(config.save_folder, 'crd_mutual.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    df = pd.read_csv('data/train.csv')
    df.dropna(subset=['picture_lists', 'content'], axis=0, inplace=True)
    df = df[df['picture_lists'].map(lambda d: d.split('.')[-1]).isin(['jpg'])]
    df['label'] = df.apply(lambda x: get_label(x), axis=1)
    df['img'] = df.apply(lambda x: get_img(x), axis=1)
    df = df[df['img'].map(lambda a: img_exist(a))]
    print(df['label'].value_counts())

    train_data, val_data = train_test_split(df, shuffle=True, test_size=0.1)
    print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))

    train_dataset = MyDataset(train_data, 'train', args.task, 1586)
    args.n_data = len(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(val_data, 'val', args.task, 1586)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    model_text = Model()
    model_img = resnet50(num_classes=3)

    module_list = nn.ModuleList([])
    module_list.append(model_img)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_crd = CRDLoss(args)

    module_list.append(criterion_crd.embed_s)
    module_list.append(criterion_crd.embed_t)
    module_list.append(model_text)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_crd)

    optimizers = []

    optimizer_img = optim.SGD(model_img.parameters(), lr=0.0003, momentum=0.9, weight_decay=5e-4)
    optimizer_text = torch.optim.Adam(model_text.parameters(), lr=2e-5)
    optimizers.append(optimizer_img)
    optimizers.append(optimizer_text)

    schedulers = []
    scheduler_img = optim.lr_scheduler.ExponentialLR(optimizer_img, gamma=0.1)
    scheduler_text = optim.lr_scheduler.ExponentialLR(optimizer_text, gamma=0.1)
    schedulers.append(scheduler_img)
    schedulers.append(scheduler_text)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    for epoch in range(1, args.epochs_num + 1):

        print("==> training...")

        time1 = time.time()
        train_crd_mutual(epoch, train_loader, module_list, criterion_list, optimizers, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        test_img_acc, test_img_loss = validate_crd(val_loader, module_list, criterion_cls, 0)
        test_text_acc, test_text_loss = validate_crd(val_loader, module_list, criterion_cls, 1)

        # save the best model
        if test_img_acc > best_img_acc:
            best_img_acc = test_img_acc
            state = {
                'epoch': epoch,
                'model': model_img.state_dict(),
                'best_acc': best_img_acc,
            }
            save_file = os.path.join(config.save_folder, 'img_best_crd_mutual.pth')
            print('saving the best model for img!')
            torch.save(state, save_file)
        if test_text_acc > best_text_acc:
            best_text_acc = test_text_acc
            state = {
                'epoch': epoch,
                'model': model_text.state_dict(),
                'best_acc': best_text_acc,
            }
            save_file = os.path.join(config.save_folder, 'text_best_crd_mutual.pth')
            print('saving the best model for text!')
            torch.save(state, save_file)
    print('best accuracy for image model :', best_img_acc)
    print('best accuracy for text model :', best_text_acc)



def train_crd_mutual(epoch, train_loader, module_list, criterion_list, optimizers, opt):
    for module in module_list:
        module.train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_img = module_list[0]
    model_text = module_list[-1]

    losses_img=AverageMeter()
    losses_text = AverageMeter()
    top1_img=AverageMeter()
    top1_text = AverageMeter()


    for idx,data in enumerate(train_loader):
        _,batch_x, batch_img, batch_y, index, contrast_idx=data
        batch_x, batch_img, batch_y = batch_x.cuda(), batch_img.cuda(), batch_y.cuda()
        index,contrast_idx=index.cuda(),contrast_idx.cuda()


        feat_s, logits_img = model_img(batch_img, is_feat=True, preact=False)
        cls_output, logits_text = model_text(batch_x)

        loss_cls_img=criterion_cls(logits_img,batch_y)
        loss_cls_text=criterion_cls(logits_text,batch_y)


        loss_div_img = criterion_div(logits_img, logits_text)
        loss_div_text = criterion_div(logits_text, logits_img)


        loss_kd = criterion_kd(feat_s[-1], cls_output, index, contrast_idx)

        train_loss_img = opt.gamma * loss_cls_img + opt.alpha * loss_div_img + opt.beta * loss_kd
        train_loss_text = opt.gamma * loss_cls_text + opt.alpha * loss_div_text + opt.beta * loss_kd

        acc1_img = accuracy(logits_img, batch_y, topk=1)
        losses_img.update(train_loss_img.item(), batch_img.size(0))
        top1_img.update(float(acc1_img[0]), batch_img.size(0))

        optimizers[0].zero_grad()
        train_loss_img.backward(retain_graph=True)
        optimizers[0].step()

        acc1_text = accuracy(logits_text, batch_y, topk=1)
        losses_text.update(train_loss_text.item(), batch_x.size(0))
        top1_text.update(float(acc1_text[0]), batch_x.size(0))

        optimizers[1].zero_grad()
        train_loss_text.backward()
        optimizers[1].step()



        if idx % config.train_print_step == 0:
            print('Image Model:')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses_img, top1=top1_img))

            print('Text Model:')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, idx, len(train_loader),
                loss=losses_text, top1=top1_text))
    print(' Train image model : Acc@1 {top1.avg:.3f}'.format(top1=top1_img))
    print(' Train text model : Acc@1 {top1.avg:.3f}'.format(top1=top1_text))
    sys.stdout.flush()




def validate_crd(val_loader, model, criterion, opt):

    """validation"""
    if opt==0:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        model_img=model[0]


        with torch.no_grad():
            cur_step = 0
            for _, _, batch_img, batch_y, _, _ in val_loader:
                batch_img, batch_y = batch_img.cuda(), batch_y.cuda()

                batch_img = batch_img.float()

                # compute output
                logits = model_img(batch_img)
                loss = criterion(logits, batch_y)

                # measure accuracy and record loss
                acc1 = accuracy(logits, batch_y, topk=1)
                losses.update(loss.item(), batch_img.size(0))
                top1.update(float(acc1[0]), batch_img.size(0))

                cur_step += 1
                if cur_step % config.train_print_step == 0:
                    print('test for image model: ')
                    print('test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        cur_step, len(val_loader), loss=losses,
                        top1=top1))

            print('val img model * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        return top1.avg,  losses.avg

    elif opt == 1:
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        model_text=model[-1]


        with torch.no_grad():
            cur_step = 0
            for _, batch_x, _, batch_y, _, _ in val_loader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                # compute output
                cls_output, logits = model_text(batch_x)
                loss = criterion(logits, batch_y)

                # measure accuracy and record loss
                acc1 = accuracy(logits, batch_y, topk=1)
                losses.update(loss.item(), batch_x.size(0))
                top1.update(float(acc1[0]), batch_x.size(0))

                cur_step += 1
                if cur_step % config.train_print_step == 0:
                    print('test for text model: ')
                    print('test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                        cur_step, len(val_loader), loss=losses,
                        top1=top1))

            print('val text model * Acc@1 {top1.avg:.3f} '.format(top1=top1))

        return top1.avg, losses.avg



if __name__ == '__main__':
    main()

#
# def predict():
#     comment_dict = {
#         0: '0c',
#         1: '2c',
#         2: 'all'
#     }
#     fake_prob_label = defaultdict(list)
#     real_prob_label = defaultdict(list)
#     ncw_prob_label = defaultdict(list)
#     test_df = pd.read_csv(config.test_path)
#     test_df.fillna({'content': ''}, inplace=True)
#
#     for i in range(3):
#         test_dataset = MyDataset(test_df, 'test', '{}'.format(i))
#         test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
#         model = resnet50(num_classes=3).to(device)
#         resume = os.path.join(config.model_path, 'img_{}_task{}_trial{}.bin'.format(model_name, i, args.trial))
#         model.load_state_dict(torch.load(resume))
#
#         model.eval()
#         with torch.no_grad():
#             for batch_x, batch_img,batch_y,_,_ in tqdm(test_loader):
#                 batch_x = batch_x.cuda
#                 logits, _ = model(batch_x)
#                 # _, preds = torch.max(probs, 1)
#                 probs = torch.softmax(logits, 1)
#                 # probs_data = probs.cpu().data.numpy()
#                 fake_prob_label[i] += [p[0].item() for p in probs]
#                 real_prob_label[i] += [p[1].item() for p in probs]
#                 ncw_prob_label[i] += [p[2].item() for p in probs]
#     submission = pd.read_csv(config.sample_submission_path)
#     for i in range(3):
#         submission['fake_prob_label_{}'.format(comment_dict[i])] = fake_prob_label[i]
#         submission['real_prob_label_{}'.format(comment_dict[i])] = real_prob_label[i]
#         submission['ncw_prob_label_{}'.format(comment_dict[i])] = ncw_prob_label[i]
#     submission.to_csv(config.submission_path + '/' + 'submission_text.csv', index=False)

