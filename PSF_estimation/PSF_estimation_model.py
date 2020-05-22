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
import math
from math import exp
import scipy.io as io
import skimage.metrics

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
		self.parser.add_argument('--lr1', type=int, default=0.00001, help='# learn rate')
		self.parser.add_argument('--beta1', type=int, default=0.5, help='# adam optimize beta1 parameter')
		self.parser.add_argument('--gpu', action='store_true', default=False, help='# use pgu')
		self.parser.add_argument('--vis', default=True, help='# wheather to use visdom visulizer')
		self.parser.add_argument('--env', type=str, default='PSF_estimation', help='# visualizer environment')
		self.parser.add_argument('--save_path', required=True, help='weight file(.pth) save path')
		self.parser.add_argument('--plot_every', type=int, default=10, help='# print error')
		self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
		self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
		self.parser.add_argument('--model', type=str, default='Train', help='chooses which model to use. test or train')	
		self.parser.add_argument('--load_model', type=str, default=None, help='# load train .pth file')
	
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
						tv.transforms.Normalize((0.1546, 0.1546, 0.1546), (0.2346, 0.2346, 0.2346))
						])

	def __getitem__(self, index):
                # image name : 25978_1blur2_20.bmp
                # 2 : blur kernel file
                # 20 : the 20th blur kernel of 2th blur kernel file      blur kernel file size : 201x39x39
		A_path = self.A_paths[index]

		A_img = Image.open(A_path).convert('L')

		A_img = self.transform(A_img)

		all_name = re.split(r'/', A_path)[-1]	
		target = re.split(r'\.', all_name)[0]
		target = re.split(r'blur', target)[-1]
		return A_img, target

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
 
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.first = nn.Sequential(
		nn.Conv2d(1, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )
        self.layer1 = self.make_layer(64, 64, 3, 1) 
        self.layer2 = self.make_layer(64, 128, 4, 2) 
        self.layer3 = self.make_layer(128, 256, 6, 2) 
        self.layer4 = self.make_layer(256, 512, 3, 2) 
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
	
	resnet = ResNet()
	if opt.load_model:
		map_location = lambda storage, loc: storage
		resnet.load_state_dict(t.load(opt.load_model, map_location=map_location))
	optimizer = t.optim.Adam(resnet.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
	criterion = t.nn.MSELoss()
	old_lr = opt.lr1

	if opt.gpu:
		resnet.cuda()
		criterion.cuda()
	Resnet34_loss_file = np.zeros([opt.max_epoch,1])
        Validation_loss_file = np.zeros([opt.max_epoch,1])

	#pre-load blur kernel file in order to accelerate training
	matfile_Dict = OrderedDict()
	for m in range(1, 835):
        	mat_file = loadmat('../lastest_train_spaceobject_dataset/crop_10_10_kernel/kernel_blur_pad/%d.mat'%(int(m)))
        	mat_file = mat_file.get('kernel_blur')
        	matfile_Dict[m] = mat_file

	for epoch in range(1, opt.max_epoch+1):
		epoch_iter = 0
		for ii,(img, labels)  in enumerate(dataloader):
			iter_start_time = time.time()
			epoch_iter += opt.batch_size
			inputs = Variable(img)
			optimizer.zero_grad()
			outputs = resnet(inputs.cuda())
			target = np.zeros((opt.batch_size, 1, 225, 225))
			for l in range(0, opt.batch_size):
				kernel_count = labels[l]
				mat = re.split(r'_', kernel_count)[0]
				kernel_blur_number = re.split(r'_', kernel_count)[1]
				kernel_blur = matfile_Dict[int(mat)]
				kernelBlur = kernel_blur[int(kernel_blur_number)]				
				target[l,:,:,:] = kernelBlur
			target = torch.Tensor(target)
			pre_outputs = Variable(target).cuda()
			loss = torch.sqrt(criterion(outputs, pre_outputs))
			loss.backward()
			optimizer.step()
			
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
                        model_save_path = os.path.join(opt.save_path, 'resnet_%s.pth'%str(epoch))
			t.save(resnet.state_dict(), model_save_path)
			Resnet34_loss_file[epoch-1,0] = loss.data
			#validation
			resnet_validata = ResNet().cuda()
			map_location = lambda storage, loc: storage
			pth = os.path.join(opt.save_path, 'resnet_%s.pth'%str(epoch))
			resnet_validata.load_state_dict(t.load(pth, map_location=map_location))
			dataset_validata = MyDataset(True)
			dataloader_validata = t.utils.data.DataLoader(dataset_validata,
                                         	batch_size=50,
                                         	shuffle=True,
                                         	num_workers=opt.num_workers,
                                         	drop_last=True
                                         	)
			for ii,(img, labels)  in enumerate(dataloader_validata):
				inputs_validata = Variable(img)
				outputs_validata = resnet_validata(inputs_validata)
				target_validata = np.zeros((50, 1, 225, 225))
				for l in range(0, 50):
					validata_kernel_count = labels[l]
					validata_mat = re.split(r'_', validata_kernel_count)[0]
					validata_kernel_blur_number = re.split(r'_', validata_kernel_count)[1]
					validata_kernel_blur = matfile_Dict[int(validata_mat)]
					validata_kernelBlur = validata_kernel_blur[int(validata_kernel_blur_number)]
					target_validata[l,:,:,:] = validata_kernelBlur
				target_validata = torch.Tensor(target_validata)
				pre_outputs_validata = Variable(target_validata)
				loss_validata = torch.sqrt(validata_criterion(outputs_validata, pre_outputs_validata))
				Validation_loss_file[epoch-1,0] = loss_validata.data
				break
		if epoch > (opt.max_epoch/2):
			lrd = opt.lr1 / (opt.max_epoch/2)
			lr = old_lr - lrd
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print('update learning rate: %f -> %f' % (old_lr, lr))
			old_lr = lr

	import scipy.io as io
	io.savemat('./every_epoch_loss/Resnet34_loss_file',{'train_loss': Resnet34_loss_file})
	io.savemat('./every_epoch_loss/Validation_loss_file',{'validation_loss': Validation_loss_file})	

def test():
	resnet = ResNet().eval()
	map_location = lambda storage, loc: storage	
	resnet.load_state_dict(t.load(opt.load_model, map_location=map_location))
	dataset = MyDataset()
	criterion = t.nn.MSELoss()
	dataloader = t.utils.data.DataLoader(dataset,
					batch_size=1,
					shuffle=False,
					num_workers=1,
					drop_last=False
					)
	psnr_data_save = np.zeros((1501, 1))
	ssim_data_save = np.zeros((1501, 1))
	total_psnr = 0
	total_ssim = 0
	for ii, (img, labels) in enumerate(dataloader):
		print('current process the %d image'%(ii))
		inputs = Variable(img)
		output = resnet(inputs)
	#	pre_output = np.zeros((1,1,225,225))
		real_kernel = labels[0]
		real_mat = re.split(r'_', real_kernel)[0]
		real_number = re.split(r'_', real_kernel)[1]
		# statistic data use crop_100_1_kernel blur kernel
		real_kernel_blur = loadmat('../lastest_train_spaceobject_dataset/crop_100_1_kernel/kernel_blur_pad/%d.mat'%(int(real_mat)))
		real_kernel_blur = real_kernel_blur.get('kernel_blur')
		real_kernelBlur = real_kernel_blur[int(real_number)]
	#	pre_output[0,0,:,:] = real_kernelBlur
	#	pre_output = Variable(torch.Tensor(pre_output))
		# ssim
		kernelBlur = output.data[0,0,:,:].numpy()
		data_ssim = skimage.metrics.structural_similarity(kernelBlur, real_kernelBlur, data_range=1)
		total_ssim = total_ssim + data_ssim
		ssim_data_save[ii, 0] = data_ssim	
		# psnr
		data_psnr = PSNR(kernelBlur, real_kernelBlur)
		total_psnr = total_psnr + data_psnr
		psnr_data_save[ii, 0] = data_psnr
		import scipy.io as io
		io.savemat('./test_kernel_blur/real_kernel' + str(labels[0]),{'real_kernel': real_kernelBlur})
		io.savemat('./test_kernel_blur/generate_kernel' + str(labels[0]),{'generate_kernel': kernelBlur})

	#average psnr ssim
	print('The dataset total number : %d'%(len(dataset)))
	average_psnr = total_psnr / len(dataset)
	print('average psnr : %.5f'%(average_psnr))
	psnr_data_save[len(dataset), 0] = average_psnr
	average_ssim = total_ssim / len(dataset)
	print('average ssim : %.5f'%(average_ssim))
	ssim_data_save[len(dataset), 0] = average_ssim
	io.savemat('./restore_sharp_psnr.mat', {'psnr_record': psnr_data_save})
	io.savemat('./restore_sharp_ssim.mat', {'ssim_record': ssim_data_save})

def PSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return Variable(window)

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1
            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity                                                     
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


def gaussian_kernel_2d(kernel_size, sigma):
	kx = cv2.getGaussianKernel(kernel_size, sigma)
	ky = cv2.getGaussianKernel(kernel_size, sigma)
	return np.multiply(kx, np.transpose(ky))

def get_current_errors(loss):
	return OrderedDict([('ResnetLoss', loss.data)])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
opt = BaseOptions().parse()
if opt.model == 'Train':
        train()
else:
        test()











































































































