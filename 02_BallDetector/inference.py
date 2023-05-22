import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import cv2
import copy

from utils import get_scheduler, get_args
from unet import UNet
from cfg import get_args

class ImgsLoader(object):

    def __init__(self, 
                 video_path: str, 
                 n_frames: int, 
                 data_size: int,
                 diff_frame: bool = False):

        self.diff_frame = diff_frame
        self.data_size = data_size
        self.n_frames = n_frames
        if self.diff_frame:
            self.n_frames += 1
        self.video_path = video_path
        files = os.listdir(video_path)
        jpg_files = [x for x in files if x.split('.')[-1] == 'jpg']
        self.frame_datas = {}
        for i in range(n_frames + 1, len(jpg_files)):
            inp_files = []
            for j in range(n_frames - 1, -1, -1):
                img_path = video_path + "/{}.jpg".format(i - j)
                assert os.path.isfile(img_path)
                inp_files.append(img_path)
            self.frame_datas[i] = inp_files

        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((data_size[1], data_size[0]))
        self.total_frames = len(self.frame_datas)

        self.buffer = self._gather_images(self.frame_datas[n_frames + 1])

    def _load_image(self, img_path):
        t = torch.from_numpy(cv2.imread(img_path))
        t = t.float() / 255
        t = t.permute(2, 0, 1) # H, W, C -> C, H, W
        return t

    def _gather_images(self, img_paths: list) -> torch.Tensor:
        images = []
        for img_path in img_paths:
            t = self._load_image(img_path)
            images.append(t)

        if self.diff_frame:
            for i in range(len(images) - 2, -1, -1):
                images[i] = images[i+1] - images[i]

        return images # [T, C, H, W]

    def getDataByFrameId(self, index: int) -> torch.Tensor:

        if index != self.n_frames + 1: # not first frame
             t = self._load_image(self.frame_datas[index][-1])
             del self.buffer[0]
             self.buffer.append(t)

        imgs = torch.stack(self.buffer)
        imgs = self.resize(imgs)
        imgs = self.normalize(imgs)
        T, C, H, W = imgs.size()
        imgs = imgs.view(T * C, H, W)
        return imgs


if __name__ == "__main__":

    args = get_args()
    
    if os.path.isdir(args.inf_save_root):
        print("Overwrite old result!")
    else:
        os.mkdir(args.inf_save_root)
    
    model = UNet(num_classes=1, in_c=12, base_c=32, n_frame=4)
    ckpt = torch.load('./last.pth')
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()

    for i in range(6, 7):
        
        vid = str(i).zfill(5)
        video_name = "val_{}".format(vid)
        
        print("process...", video_name)

        test_folder = "../data/entire_frame/{}/".format(video_name)
        loader = ImgsLoader(test_folder, 4, [args.inW, args.inH])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outv = cv2.VideoWriter('./{}/{}.mp4'.format(args.inf_save_root, video_name), fourcc, 30.0, (1280,  720), isColor=1)
        msg = ""
        with torch.no_grad():
            for i in range(4 + 1, loader.total_frames):
                imgs = loader.getDataByFrameId(i)
                imgs = imgs.unsqueeze(0)
                imgs = imgs.cuda()
                outs = model(imgs)
                outs = outs.cpu()
                score = outs.sigmoid().max()
                vis = outs[0][0]

                H, W = args.inH, args.inW
                f = vis.flatten(0)
                
                max_idx = torch.max(f, dim=0)[1]
                max_idx = int(max_idx)
                
                max_row = int(max_idx / W)
                max_col = max_idx - max_row * W

                ori_arr = cv2.imread(test_folder + "{}.jpg".format(i))
                outv.write(ori_arr)
                _max_row = max_row * (1280 / args.inW)
                _max_col = max_col * (720 / args.inH)
                msg += '{} {} {}\n'.format(_max_row, _max_col, float(score))

        with open('./{}/{}.txt'.format(args.inf_save_root, video_name), "w") as ftxt:
            ftxt.write(msg)
        
        outv.release()
