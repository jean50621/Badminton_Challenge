import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
import cv2
from cfg import get_args

class ImagesPathLoader(object):

    def __init__(self, video_root: str, yolo_net_root: str, data_size: list = [480, 320]):
        """
        video_root: contain frame images.
        yolo_net_root: contain net location txt file.
        """
        self.video_root = video_root
        self.yolo_net_root = yolo_net_root
        self.frame_paths = self.getDataInfo()
        # ==== transforms ====
        self.resize = transforms.Resize((data_size[0], data_size[1]))
        self.transforms =  transforms.Compose([
            transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        ])

    def __len__(self):
        return len(self.frame_paths)


    def load_net_loc(self, obj_path: str):
        if not os.path.isfile(obj_path):
            return None, None

        with open(obj_path, "r") as ftxt:
            datas = ftxt.read().split('\n')

        bboxs = []
        for data in datas:
            if data == '':
                continue
            info = data.split(' ')
            if info[0] != '1':
                continue
            bboxs.append([float(x) for x in info[1:]])
        

        if len(bboxs) == 0:
            return None, None
        elif len(bboxs) == 1:
            x = bboxs[0][0]
            w = bboxs[0][2]
            return x - 0.5 * w, x + 0.5 * w
        else:
            max_i, max_area = None, None
            for i in range(len(bboxs)):
                a = bboxs[i][2] * bboxs[i][3]
                if max_area is None or a > max_area:
                    max_area = a
                    max_i = i
            x = bboxs[i][0]
            w = bboxs[i][2]
            return x - 0.5 * w, x + 0.5 * w
        

    def getDataInfo(self):
        """
        video_root/
            0.jpg
            1.jpg
            2.jpg
            ...
        yolo_net_root/
            1.txt
            2.txt
            ...
        (The number of file name is corresponding to the minus one in video_root)
        """
        _files = os.listdir(self.video_root)
        files = []
        for f in _files:
            ext = f.split('.')[-1]
            if ext.lower() not in ['jpg']:
                continue
            files.append(f)
        frame_paths = {}
        for i in range(3, len(files)):
            save_this_frame = True
            xbounds = []
            img_paths = []
            for j in range(4):
                idx = i - (3 - j)
                img_name = "{}.jpg".format(idx)
                if img_name not in files:
                    raise NameError("Frame {} not found in {}.".format(i, self.video_root))
                net_txt_path = self.yolo_net_root + "{}.txt".format(idx + 1)
                xl, xr = self.load_net_loc(net_txt_path)
                if xl is None:
                    save_this_frame = False
                    break
                img_paths.append(self.video_root + img_name)
                xbounds.append([xl, xr])

            if save_this_frame:
                frame_paths[i] = {
                    "image_paths": img_paths, 
                    "xbounds": xbounds
                }
        return frame_paths

    def loadClipByFrameIdx(self, frame_id: str):
        if frame_id not in self.frame_paths:
            return None
        img_paths = self.frame_paths[frame_id]['image_paths']
        xbounds = self.frame_paths[frame_id]['xbounds']
        video = []
        for i in range(4):
            arr = cv2.imread(img_paths[i])
            xl, xr = xbounds[i]
            W = arr.shape[1]
            xl = int(W * xl)
            xr = int(W * xr)
            arr = arr[:, xl:xr, :]
            t = torch.from_numpy(arr)
            t = t.float()
            t = t / 255.0
            t = t.permute(2, 0, 1)
            t = self.resize(t)
            video.append(t)
        video = torch.stack(video) # T, C, H, W
        video = self.transforms(video)
        video = video.permute(1, 0, 2, 3) 
        return video




if __name__ == "__main__":

    args = get_args()


    ### step1 build model
    model = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=True)
    model.blocks[5].pool.pool = torch.nn.AvgPool3d(kernel_size=(4, 10, 10), stride=1, padding=0)
    model.blocks[5].proj = torch.nn.Linear(args.in_features, args.out_features, bias=True)
    ckpt = torch.load(args.weight)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    
    pbar = tqdm(total=1, ascii=True)
    for i in range(6, 7):
        
        idx = str(i).zfill(5)
        video_name = '{}_{}'.format(args.prefix, idx)
        video_root = '{}/{}/'.format(args.frame_root,video_name)
        # yolo_net_root = '../part2_loc_filtered/{}/'.format(video_name) # for train and validation set.
        yolo_net_root = '{}/{}/'.format(args.yolo_net_root,video_name)
        if not os.path.isdir(yolo_net_root):
            pbar.update(1)
            continue
        
        results_root = './result_hitters_{}_last_pth/'.format(args.prefix)
        os.makedirs(results_root, exist_ok=True)
        
        pbar.set_description("process video: {}".format(video_name))

        loader = ImagesPathLoader(video_root, yolo_net_root, args.data_size)
        # print("Loader length: ", len(loader))
        imgs_list = [x for x in os.listdir(video_root) if x.split('.')[-1] == 'jpg']
        video_length = len(imgs_list)
        # print("Video length:", video_length)

        ## video writer
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # vwriter = cv2.VideoWriter(results_root + video_name + ".mp4", fourcc, 30.0, (1280, 720))

        cls_table = {
            0:"A",
            1:"B",
            2:"None"
        }
        msg = ""
        for i in range(video_length):
            frames = loader.loadClipByFrameIdx(i)
            if frames is None:
                # print('Missing frame {}'.format(i))
                continue

            # ==== forward ====
            frames = frames.unsqueeze(0).cuda() # add batch and put it on GPU
            out = model(frames)
            # ==== extract result ====
            out = out.cpu()
            score, pred = torch.max(torch.softmax(out[0], dim=0), dim=0)
            pred = int(pred)
            score = round(float(score) * 100, 3)

            msg += "{} {} {}\n".format(i, cls_table[pred], score)

        with open(results_root + video_name + ".txt", "w") as ftxt:
            ftxt.write(msg)

        pbar.update(1)

    pbar.close()



