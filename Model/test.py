#This code is used separately to test
#Input and output paths need to be modified
import cv2
import numpy as np
import torchvision.utils as vutils
from utils import decode_segmap
from SegDataFolder import semData
from getSetting import get_yaml, get_criterion, get_optim, get_scheduler, get_net
from tensorboardX import SummaryWriter
from modeling.deeplab import *
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import Dataset
import torch.nn as nn
import torch



class Solver(object):
    def __init__(self, configs):
        self.configs = configs
        self.cuda = torch.cuda.is_available()
        self.n_classes = self.configs['n_classes']
        self.ignore_index = self.configs['ignore_index']
        self.channels = self.configs['channels']
        self.net = get_net(self.configs)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = get_criterion(self.configs)
        self.optimizer = get_optim(self.configs, self.net)
        self.scheduler = get_scheduler(self.configs, self.optimizer)
        self.batchsize = self.configs['batchsize']
        self.start_epoch = self.configs['start_epoch']
        self.end_epoch = self.configs['end_epoch']
        self.logIterval = self.configs['logIterval']
        self.valIterval = self.configs['valIterval']

        self.resume = self.configs['resume']['flag']
        if self.resume:
            self.resume_state(self.configs['resume']['state_path'])
        
        if self.cuda:
            self.net = self.net.cuda()
            if self.resume:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        self.trainSet = semData(train=True, channels=self.channels)
        self.valSet = semData(train=False, channels=self.channels)
        self.train_dataloader = torch.utils.data.DataLoader(self.trainSet, batch_size=self.batchsize, shuffle=True) 
        self.val_dataloader = torch.utils.data.DataLoader(self.valSet, 1, shuffle=False)

        self.best_miou = 0.00
        self.result_dir = self.configs['result_dir']
        os.makedirs(self.result_dir,exist_ok=True)
        self.writer = SummaryWriter(self.result_dir)
        with self.writer:
            if not self.resume:
                inp = torch.randn([1,self.channels,256,256]).cuda() if self.cuda else torch.randn([1,self.channels,256,256])
                self.writer.add_graph(self.net, inp)

    def save_state(self, epoch, path):
        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, path)

    def resume_state(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        print('Resume from epoch{}...'.format(self.start_epoch))

    def test(self, singleImages, outName):
        assert len(singleImages) == self.channels
        from PIL import Image
        import transform as T
        mean = np.array([0.01308804, -1.1662938e-05, -3.957561e-05, 0.0026290491, 20.119514, 0.6648699, 0.62416124, 5.9405203, 2.0423028])
        std = np.array([0.21623433, 0.01678721, 0.012647983, 0.005536756, 7.823991, 0.14901447, 0.16086595, 1.3160306, 14.86331])

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
                return list(range(7))[-2:]

        mean_, std_ = mean[get_idx(self.channels)], std[get_idx(self.channels)]
        _t = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean_, std=std_)
        ])
        L = []
        for item in singleImages:
            img = Image.open(item)
            img = np.expand_dims(np.array(img), axis=2)
            L.append(img)
        image = np.concatenate(L, axis=-1)
        img, _ = _t(image, image[:, :, 0])
        img = img.unsqueeze(0)
        img = img.cuda() if self.cuda else img

        self.net.eval()
        with torch.no_grad():
            outp = self.net(img)
            score = self.softmax(outp)
            pred = score.max(1)[1]
            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 255 * saved_1
            cv2.imwrite(r'D:\NET1\All_Net_GN\Net\testresult\1\{}.png'.format(outName), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(r'D:\NET1\All_Net_GN\Net\testresult\255\{}.png'.format(outName), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    def init_logfile(self):
        self.vallosslog = open(os.path.join(self.result_dir,'valloss.csv'),'w')
        self.vallosslog.writelines('epoch,loss\n')   

        self.valallacclog = open(os.path.join(self.result_dir,'valacc.csv'),'w')
        self.valallacclog.writelines('epoch,acc\n')

        self.trainlosslog = open(os.path.join(self.result_dir,'trainloss.csv'),'w')
        self.trainlosslog.writelines('epoch,loss\n')

        self.trainallacclog = open(os.path.join(self.result_dir,'trainacc.csv'),'w')
        self.trainallacclog.writelines('epoch,acc\n')

        self.precisionlog = open(os.path.join(self.result_dir,'presion.csv'),'w')
        self.precisionlog.writelines('epoch,precision\n')

        self.recalllog = open(os.path.join(self.result_dir,'recall.csv'),'w')
        self.recalllog.writelines('epoch,recall\n')

        self.f1log = open(os.path.join(self.result_dir, 'f1.csv'),'w')
        self.f1log.writelines('epoch,f1\n')

    def close_logfile(self):
        self.vallosslog.close()
        self.valallacclog.close()
        self.trainlosslog.close()
        self.trainallacclog.close()
        self.precisionlog.close()
        self.recalllog.close()
        self.f1log.close()  

    def trainer(self):
        try:
            for _ in range(self.start_epoch):
                self.scheduler.step()
            
            for epoch in range(self.start_epoch, self.end_epoch):
                self.train(epoch)
                self.scheduler.step()
                self.save_state(epoch, '{}/{}-ep{}.pth'.format(self.result_dir,self.configs['net'], epoch))
                if (epoch+1)%self.valIterval == 0:
                    self.val(epoch)
        except KeyboardInterrupt:
            print('Saving checkpoints from keyboardInterrupt...')
            self.save_state(epoch, '{}/{}-kb_resume.pth'.format(self.result_dir,self.configs['net']))
        finally:
            self.writer.close()

    def visualize(self, img, label, pred):
        label = label.clone().squeeze(0).cpu().numpy()
        label = decode_segmap(label).transpose((2,0,1))
        label = torch.from_numpy(label).unsqueeze(0)
        label = label.cuda() if self.cuda else label
        pred = pred.clone().squeeze(0).cpu().numpy()
        pred = decode_segmap(pred).transpose((2,0,1))
        pred = torch.from_numpy(pred).unsqueeze(0)
        pred = pred.cuda() if self.cuda else pred
        vis = torch.cat([self.denorm(img), label.float(), pred.float()], dim=0)
        vis_cat = vutils.make_grid(vis,nrow=3,padding=5,pad_value=0.8)
        return vis_cat

    def denorm(self, x):
        mean_ = torch.Tensor().view(3,1,1)
        std_ = torch.Tensor().view(3,1,1)
        mean_ = mean_.cuda() if self.cuda else mean_
        std_ = std_.cuda() if self.cuda else std_
        out = x * std_ + mean_
        out = out / 255.
        return out.clamp_(0,1)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = get_yaml('ConfigFiles/config-deeplab.yaml')
    print(config)
    labpath = r'E:\Deeplab_Net\Net\Data\eval\images\1-C11'
    path = os.path.dirname(labpath)
    solver = Solver(config)
    labdir = os.listdir(labpath)
    print(labdir)
    for i in range(len(labdir)):
        img_path = []
        stri = labdir[i].split('.')[0] + '.tif'
        img_path.append(path + '/1-C11/' + stri)
        img_path.append(path + '/2-C12_real/' + stri)
        img_path.append(path + '/3-C12_imag/' + stri)
        img_path.append(path + '/4-C22/' + stri)
        img_path.append(path + '/5-alpha/' + stri)
        img_path.append(path + '/6-anisotropy/' + stri)
        img_path.append(path + '/7-entropy/' + stri)
        img_path.append(path + '/8-wind_speed/' + stri)
        img_path.append(path + '/9-rain/' + stri)
        solver.test(img_path, outName=stri.split()[0])

