import os
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
from SegDataFolder import semData
from getSetting import get_yaml, get_net
from metric import AverageMeter, intersectionAndUnion
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import time

mean = np.array([0.01308804, -1.1662938e-05, -3.957561e-05, 0.0026290491, 20.119514, 0.6648699, 0.62416124, 5.9405203, 2.0423028])       # 训练原图的平均值
std = np.array([0.21623433, 0.01678721, 0.012647983, 0.005536756, 7.823991, 0.14901447, 0.16086595, 1.3160306, 14.86331])  # 训练原图的方差

def get_idx(channels):
    assert channels in [2, 4, 7, 8, 9]
    if channels == 7:
        return list(range(7))
    elif channels == 8:
        return list(range(8))
    elif channels == 9:
        return list(range(9))
    elif channels == 4:
        return list(range(4))
    elif channels == 2:
        return list(range(6))[-2:]

def getTransorm4eval(channel_idx=[0,1,2,3]):
    import transform as T
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean[channel_idx],std=std[channel_idx])
        ]
    )

def eval(args,config,):
    os.makedirs(args.outputdir, exist_ok=True)
    net = get_net(config) 
    softmax = nn.Softmax(dim=1)
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda() if args.use_gpu else net
    dataset = semData(
        train=False,
        root='D:/NET1/All_Net_GN/Net/Data',
        channels = config['channels'],
        transform=getTransorm4eval(channel_idx=get_idx(config['channels'])),
        selftest_dir=args.testdir
    )
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    net.eval()
    with torch.no_grad():
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        correct = []
        preds = []
        for i, batch in enumerate(dataloader):
            img = batch['X'].cuda() if args.use_gpu else batch['X']
            label = batch['Y'].cuda() if args.use_gpu else batch['Y']
            path = batch['path'][0].split('.')[0]
            outp = net(img)
            score = softmax(outp)      
            pred = score.max(1)[1]
            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 255 * saved_1
            cv2.imwrite('eval_output/eval_output_1/{}.png'.format(path), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite('eval_output/eval_output_255/{}.png'.format(path), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            correct.extend(label.reshape(-1).cpu().numpy())
            preds.extend(pred.reshape(-1).cpu().numpy())
            N = img.size()[0]
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), label.cpu().numpy(), config['n_classes'], config['ignore_index'])
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            acc = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(acc_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum)+1e-10)
        precision = precision_score(correct, preds)
        recall = recall_score(correct, preds)
        f1 = f1_score(correct, preds)

        print('iou_0: {:.4f} | iou_1: {:.4f} | mIoU {:.4f} | mAcc {:.4f} | allAcc {:.4f} | precision {:.4f} | recall {:.4f} | f1 {:.4f}'.format(
            iou_class[0],iou_class[1],mIoU, mAcc, allAcc, precision, recall, f1
        ))

parser = argparse.ArgumentParser()
parser.add_argument('--testdir',type=str,default='eval', help='path of test images')
parser.add_argument('--outputdir',type=str,default='eval_output',help='test output')
parser.add_argument('--gpu',type=int, default=0,help='gpu index to run')
parser.add_argument('--configfile',type=str, default='ConfigFiles/config-deeplab.yaml',help='path to config files')
parser.add_argument('--checkpoint',type=str, default=r'F:\deeplabv3+训练模型\eventfiles-DeepLab-channels9\DeepLab-ep57.pth', help='checkpoint path') # 加载定模型路径
args = parser.parse_args()

if __name__ == "__main__":
    t1 = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False
    config = get_yaml(args.configfile)
    eval(args, config)
    t2 = time.time()
    print(t2-t1)