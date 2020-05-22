import argparse
import torch as t
import torch
import torch.nn as nn
import numpy as np
import re
import torchvision as tv
from torch.autograd import Variable
import torch.nn.functional as F
import os
from PIL import Image
import cv2
from collections import OrderedDict
import time
from scipy.io import loadmat


#define argument
class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--data_path', required=True, help='path to images')
		self.parser.add_argument('--validata_path', required=True, help='path to validate images')
		self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
		self.parser.add_argument('--num_workers', default=0, type=int, help='# threads for loading data')
		self.parser.add_argument('--image_size', type=int, default=225, help='then crop to this size')
		self.parser.add_argument('--max_epoch', type=int, default=300, help='# epoch count')
		self.parser.add_argument('--lr1', type=int, default=0.0000001, help='# learn rate')
		self.parser.add_argument('--beta1', type=int, default=0.5, help='# adam optimize beta1 parameter')
		self.parser.add_argument('--gpu', action='store_true', default=False, help='# use pgu')
		self.parser.add_argument('--vis', default=True, help='# wheather to use visdom visulizer')
		self.parser.add_argument('--env', type=str, default='blind_deconvolution', help='# visualizer environment')
		self.parser.add_argument('--save_path', required=True, help='weight file(.pth) save path')
		self.parser.add_argument('--plot_every', type=int, default=10, help='# print error')
		self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
		self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
		self.parser.add_argument('--model', type=str, default='Train', help='chooses which model to use. test or train')	
		self.parser.add_argument('--resnet1_path', default=None, help='wheather load PSF estimation network for fine-turning')
		self.parser.add_argument('--resnet2_path', default=None, help='wheather load non-blind deconvolution network for fine-turning')
	
	def parse(self):
                if not self.initialized:
                        self.initialize()
                self.opt = self.parser.parse_args()
                args = vars(self.opt)

                print('------------ Options -------------')
                for k, v in sorted(args.items()):
                        print('%s: %s' % (str(k), str(v)))
                print('-------------- End ----------------')
                return self.opt

#define dataset
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, validata=False):
		self.opt = opt
		if  validata:
			self.root = opt.validata_path
			self.dir_A = os.path.join(opt.validata_path)
		else:
			self.root = opt.data_path
			self.dir_A = os.path.join(opt.data_path)

		self.A_paths = make_dataset(self.dir_A)

		self.A_paths = sorted(self.A_paths)

		self.transform = tv.transforms.Compose([
						tv.transforms.Resize(opt.image_size),
						tv.transforms.CenterCrop(opt.image_size),
						tv.transforms.ToTensor(),
						tv.transforms.Normalize((0.1611, 0.1611, 0.1611), (0.2381, 0.2381, 0.2381))
						])

	def __getitem__(self, index):
		A_path = self.A_paths[index]

		A_img = Image.open(A_path).convert('L')

		A_img = self.transform(A_img)

		all_name = re.split(r'/', A_path)[-1]	
		
		sharp_img = re.split(r'blur', all_name)[0]
		return A_img, sharp_img, all_name

	def __len__(self):
		return len(self.A_paths)	


#define network

class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            if in_channels != out_channels:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                        nn.BatchNorm2d(out_channels)
                        )
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)


            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out
 
class ResNet1_psf(nn.Module):
    def __init__(self):
        super(ResNet1_psf, self).__init__()
        self.first = nn.Sequential(
		nn.Conv2d(1, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        self.layer1 = self.make_layer(64, 64, 3, 1) #in_channel=64,out_channel=64,block_num=3,stride=1
        self.layer2 = self.make_layer(64, 128, 4, 2) #in_channel=64,out_channel=128,block_num=4,stride=2
        self.layer3 = self.make_layer(128, 256, 6, 2) #in_channel=128,out_channel=256,block_num=6,stride=2
        self.layer4 = self.make_layer(256, 512, 3, 2) #in_channel=256,out_channel=512,block_num=3,stride=2
        self.second = nn.Sequential(
                nn.Conv2d(512, 512, 8, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                )
        self.fc = nn.Linear(512, 50625)
        self.end = nn.Sequential(
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.ReplicationPad2d(1),
               nn.Conv2d(512, 256, 4, 1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.ReplicationPad2d(1),
               nn.Conv2d(256, 128, 4, 1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.ReplicationPad2d(1),
               nn.Conv2d(128, 64, 4, 1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.ReplicationPad2d(1),
               nn.Conv2d(64, 32, 4, 1),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.ReplicationPad2d(1),
               nn.Conv2d(32, 1, 4, 1),
        )
        self.conv1 = nn.Conv2d(2, 512, 1, 1)
        self.conv2 = nn.Conv2d(512, 1, 1, 1)

    def make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.second(x)
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.fc(x1)
        x1 = x1.view(x1.size()[0], 1, 225, 225)
        x2 = self.end(x)
        x = torch.cat((x1,x2), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class ResNet2_dec(nn.Module):
        def __init__(self):
            super(ResNet2_dec, self).__init__()
            self.first = nn.Sequential(
                            nn.Conv2d(2, 64, 7, 2, 3),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, 2, 1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True)
                            )
            self.layer1 = self.make_layer(64, 64, 3, 1) #in_channel=64,out_channel=64,block_num=3,stride=1
            self.layer2 = self.make_layer(64, 128, 4, 2) #in_channel=64,out_channel=128,block_num=4,stride=2
            self.layer3 = self.make_layer(128, 256, 6, 2) #in_channel=128,out_channel=256,block_num=6,stride=2
            self.layer4 = self.make_layer(256, 512, 3, 2) #in_channel=256,out_channel=512,block_num=3,stride=2
            self.second = nn.Sequential(
                nn.Conv2d(512, 512, 8, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                )
            self.fc = nn.Linear(512, 50625)
            self.end = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(512, 256, 4, 1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(256, 128, 4, 1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(128, 64, 4, 1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(64, 32, 4, 1),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='nearest'),
                            nn.ReplicationPad2d(1),
                            nn.Conv2d(32, 1, 4, 1),
                            )
            self.conv1 = nn.Conv2d(2, 512, 1, 1)
            self.conv2 = nn.Conv2d(512, 1, 1, 1)

        def make_layer(self, in_channels, out_channels, block_num, stride):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))

            for i in range(1, block_num):
                layers.append(ResidualBlock(out_channels, out_channels, 1))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.first(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x1 = self.second(x)
            x1 = x1.view(x1.size()[0], -1)
            x1 = self.fc(x1)
            x1 = x1.view(x1.size()[0], 1, 225, 225)
            x2 = self.end(x)
            x = torch.cat((x1,x2), 1)
            x = self.conv1(x)
            x = self.conv2(x)
            x = F.sigmoid(x)
            return x

# visulize
class Visualizer():
        def __init__(self, opt):
                self.display_id = opt.display_id
                self.win_size = opt.display_winsize
                self.name = 'experiment_name'
                if self.display_id:
                        import visdom
                        self.vis = visdom.Visdom(env=opt.env, port=opt.display_port)
                        self.display_single_pane_ncols = opt.display_single_pane_ncols

        def plot_current_errors(self, epoch, count_ratio, opt, errors):
                if not hasattr(self, 'plot_data'):
                        self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
                self.plot_data['X'].append(epoch + count_ratio)
                self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
                self.vis.line(
                        X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1) if len(self.plot_data['X'])==1 else np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1).squeeze(),
                        Y=np.array(self.plot_data['Y']) if len(self.plot_data['Y'])==1 else np.array(self.plot_data['Y']).squeeze(),
                        opts={
                                'title': self.name + ' loss over time',
                                'legend': self.plot_data['legend'],
                                'xlabel': 'epoch',
                                'ylabel': 'loss'},
                        win=self.display_id)

        def print_current_errors(self, epoch, i, errors, t):
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
                for k, v in errors.items():
                        message += '%s: %.8f ' % (k, v)
                print(message)

#train network
def train():
	if opt.vis:
		vis  = Visualizer(opt)
	
	dataset = MyDataset()
	dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )
	
	resnet1_psf = ResNet1_psf()
	resnet2_dec = ResNet2_dec()	
	map_location = lambda storage, loc: storage
	if opt.resnet1_path:
            resnet1_psf.load_state_dict(t.load(opt.resnet1_path, map_location=map_location))
	if opt.resnet2_path:
            resnet2_dec.load_state_dict(t.load(opt.resnet2_path, map_location=map_location))

	optimizer_psf = t.optim.Adam(resnet1_psf.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
	optimizer_dec = t.optim.Adam(resnet2_dec.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
	criterion = t.nn.MSELoss()
	old_lr = opt.lr1

	if opt.gpu:
		resnet1_psf.cuda()
		resnet2_dec.cuda()
		criterion.cuda()

	Resnet34_loss_file = np.zeros([opt.max_epoch,1])	
        Validation_loss_file = np.zeros([opt.max_epoch,1])

	for epoch in range(1, opt.max_epoch+1):
		epoch_iter = 0
		for ii,(img, labels, _)  in enumerate(dataloader):
			iter_start_time = time.time()
			epoch_iter += opt.batch_size
			inputs_psf = Variable(img).cuda()
			optimizer_psf.zero_grad()
			optimizer_dec.zero_grad()
			outputs_psf = resnet1_psf(inputs_psf)
			inputs_dec = torch.cat((inputs_psf, outputs_psf), 1)
			outputs_dec = resnet2_dec(inputs_dec)                        
			target = np.zeros((opt.batch_size, 1, 225, 225))
			for l in range(0, opt.batch_size):
				sharp_img = labels[l]
				tmp_img = '%s.bmp'%(sharp_img)
				tmp = sharpimg_Dict[tmp_img]
				target[l,0,:,:] = tmp
			target = torch.Tensor(target)
			pre_outputs = Variable(target).cuda()
			loss = torch.sqrt(criterion(outputs_dec, pre_outputs))
			loss.backward()
			optimizer_psf.step()
			optimizer_dec.step()

			if opt.vis and (ii+1)%opt.plot_every == 0:
				errors = get_current_errors(loss)
				ti = (time.time() - iter_start_time) / opt.batch_size
				vis.print_current_errors(epoch, epoch_iter, errors, ti)
				with open('Resnet34.txt','a') as f:
                                        vdl = 'epoch:%d Resnet34_loss:%.10f'%(epoch, loss)
                                        f.write(vdl + '\n')
                                        f.close()

			if opt.display_id > 0 and (ii+1)%100 == 0:
				load_rate = float(epoch_iter)/dataset.__len__()
				vis.plot_current_errors(epoch, load_rate, opt, errors)	
				
		if epoch%1 == 0:
                        model_save_path_psf = os.path.join(opt.save_path, 'resnet_%s_psf.pth' % str(epoch))
                        model_save_path_dec = os.path.join(opt.save_path, 'resnet_%s_dec.pth' % str(epoch))
			t.save(resnet1_psf.state_dict(), model_save_path_psf)
			t.save(resnet2_dec.state_dict(), model_save_path_dec)
			Resnet34_loss_file[epoch-1,0] = loss.data
			#validation
			resnet1_validata_psf = ResNet1_psf().cuda()
                        resnet2_validata_dec = ResNet2_dec().cuda()
                        map_location = lambda storage, loc: storage
                        resnet1_psf.load_state_dict(t.load(model_save_path_psf, map_location=map_location))
                        resnet2_dec.load_state_dict(t.load(model_save_path_dec, map_location=map_location))
			dataset_validata = MyDataset(True)
			dataloader_validata = t.utils.data.DataLoader(dataset_validata,
                                        	batch_size=50,
                                         	shuffle=True,
                                         	num_workers=opt.num_workers,
                                         	drop_last=True
                                         	)
			for ii,(img, labels, _)  in enumerate(dataloader_validata):
                                inputs_validata_psf = Variable(img).cuda()
                                outputs_validata_psf = resnet1_validata_psf(inputs_validata_psf)
                                inputs_validata_dec = torch.cat((inputs_validata_psf, outputs_validata_psf), 1)
                                outputs_validata_dec = resnet2_validata_dec(inputs_validata_dec)		
                		target_validata = np.zeros((50, 1, 225, 225))
				for l in range(0, 50):
					validata_sharp_img = labels[l]
                                        tmp_validata_sharp_img = '%s.bmp'%(validata_sharp_img)
                                        tmp_validata_sharpImg = sharpimg_Dict[tmp_validata_sharp_img]
					target_validata[l,:,:,:] = tmp_validata_sharpImg
				target_validata = torch.Tensor(target_validata)
				pre_outputs_validata = Variable(target_validata).cuda()
				loss_validata = torch.sqrt(criterion(outputs_validata, pre_outputs_validata))
				Validation_loss_file[epoch-1,0] = loss_validata.data

	import scipy.io as io
	io.savemat('./every_epoch_loss/Resnet34_loss_file',{'train_loss': Resnet34_loss_file})
        io.savemat('./every_epoch_loss/Validation_loss_file',{'validation_loss': Validation_loss_file})

def test():
	resnet1 = ResNet1_psf().eval()
	resnet2 = ResNet2_dec().eval()
	map_location = lambda storage, loc: storage
	resnet1.load_state_dict(t.load(opt.resnet1_path, map_location=map_location))
	resnet2.load_state_dict(t.load(opt.resnet2_path, map_location=map_location))	
	dataset = MyDataset()
	criterion = t.nn.MSELoss()
	dataloader = t.utils.data.DataLoader(dataset,
					batch_size=1,
					shuffle=False,
					num_workers=1,
					drop_last=False
					)
	for ii, (img, labels, all_name) in enumerate(dataloader):
		inputs = Variable(img)
		outputs_psf = resnet1(inputs)
		inputs_dec = torch.cat((inputs, outputs_psf), 1)
		outputs_dec = resnet2(inputs_dec)
		output_numpy = outputs_dec.data[0,0,:,:].cpu().float().numpy()
		output_numpy = output_numpy * 255.0
		output_numpy = output_numpy.astype(np.uint8)
		output_PIL = Image.fromarray(output_numpy, mode='L')
		output_PIL.save('./test_all_img/res%s'%(all_name[0]))

def gaussian_kernel_2d(kernel_size, sigma):
	kx = cv2.getGaussianKernel(kernel_size, sigma)
	ky = cv2.getGaussianKernel(kernel_size, sigma)
	return np.multiply(kx, np.transpose(ky))

def get_current_errors(loss):
	return OrderedDict([('ResnetLoss', loss.data)])

# pre load all sharp image in order to accelerate training
sharpimg_Dict = OrderedDict()
sharp_img = os.listdir('/workspace/ysb/lastest_train_spaceobject_dataset/crop_100_1_kernel/source_img')
sharp_img_list = []
for si in sharp_img:
        sharp_img_path = os.path.join('/workspace/ysb/lastest_train_spaceobject_dataset/crop_100_1_kernel/source_img', si)
        sharp_img_list.append(sharp_img_path)
for sil in sharp_img_list:
        sharpimg_name = re.split(r'/', sil)[-1]
        sharpimg = Image.open(sil).convert('L')
        sharpimg = np.array(sharpimg)/255
        sharpimg_Dict[sharpimg_name] = sharpimg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
opt = BaseOptions().parse()
if opt.model == 'Train':
        train()
else:
        test()











































































































