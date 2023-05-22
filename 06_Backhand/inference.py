import torch
import torchvision.transforms as transforms
import torch.nn as nn
import csv
import os
import cv2
import tqdm
from pathlib import Path
from cfg import get_args

class DataLoader(object):

    def __init__(self,
                 entri_frames_root: str, 
                 yolo_pred_root: str,
                 gt_root: str,
                 data_size: int,
                 n_frames: int = 5):
        self.target_goal = 'Backhand'
        self.entri_frames_root = entri_frames_root
        self.yolo_pred_root = yolo_pred_root
        self.gt_root = gt_root
        self.data_size = data_size
        self.n_frames = n_frames

        self.resize = transforms.Resize((data_size[0], data_size[1]))

        self.transforms =  transforms.Compose([
            transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        ])

    def load_csv(self, csv_path):

        if self.target_goal == 'RoundHead':
            read_col = 3
        elif self.target_goal == 'Backhand':
            read_col = 4
        
        rows = []
        
        with open(csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)


        frames_id = []
        labels = {}
        for r in rows[1:]:
            pivot = int(r[1])
            if pivot-3 <0 :
                 frames_id.append(0)
                 labels[0] = [int(r[read_col]) - 1, r[2]] # 1, 2 --> 0, 1
            else :
                frames_id.append(pivot-3)
                labels[pivot-3] = [int(r[read_col]) - 1, r[2]] # 1, 2 --> 0, 1

        return frames_id, labels

    def process_player_loc(self,obj_path: str, now_frame:int):

        obj_paths = os.listdir(obj_path)
        new_players=[]

        for index in range(now_frame-1,1,-1):

            if "{}.txt".format(index) not in obj_paths:
                continue
            
            open_path = obj_path + "/{}.txt".format(index)
                
            with open(open_path, "r") as ftxt:
                players=[]
                datas = ftxt.read().split('\n')
                for data in datas:
                    if data == '':
                        continue
                    info = data.split(' ')
                    if info[0] != '0':
                        continue
                    x, y, w, h = [float(x) for x in info[1:]]
                    players.append([x, y, w, h])

                if len(players) != 2:
                    continue
                else :
                    new_players = players
                    break

        return new_players
    
    def load_player_loc(self, obj_path: str, obj_folder: str, now_frame: int):

        with open(obj_path, "r") as ftxt:
            datas = ftxt.read().split('\n')

        players = []
        for data in datas:
            if data == '':
                continue
            info = data.split(' ')
            if info[0] != '0':
                continue
            x, y, w, h = [float(x) for x in info[1:]]
            players.append([x, y, w, h])

        
        result = {
            'A':[None, None, None, None],
            'B':[None, None, None, None]
        }

        if len(players) <2 :
            
            players = self.process_player_loc(obj_folder, now_frame)
            
        
        if len(players) ==3:
            del players[0]


        if players[0][1] < players[1][1]:
            result['A'] = players[0]
            result['B'] = players[1]
        else:
            result['A'] = players[1]
            result['B'] = players[0]

        return result

    def loadVideo(self, folder: str):
        """
        folder: 00001, 00002...
        """
        print("video:",folder)
        obj_path = self.yolo_pred_root + "val_" + folder 
        img_path = self.entri_frames_root + "val_" + folder
        gt_path = self.gt_root + "/" + folder + ".csv"
        
        choose_frames_id, labels = self.load_csv(gt_path)

        choose_frames_id = set(choose_frames_id)
        print(choose_frames_id)

        img_paths = [x for x in os.listdir(img_path) if x.split('.')[-1] == 'jpg']  
        obj_paths = os.listdir(obj_path) # need to minus one

        _datas = {}
        for i in range(len(img_paths)):
            img_name = "{}.jpg".format(i)
            obj_name = "{}.txt".format(i + 1)

            if obj_name not in obj_paths:
                for k in range(i+1, 1, -1):
                    new_obj = "{}.txt".format(k)
                    if new_obj in obj_paths:
                        obj_name = new_obj
                        break
                    else : 
                        continue

            bbox_info = self.load_player_loc(obj_path + "/" + obj_name,obj_path, i+1)
            if bbox_info == None:
                continue
            _datas[i] = {
                "path": img_path + "/" + img_name,
                "bbox":bbox_info
            }


        data_infos = []
            
        for i in range(len(img_paths)-self.n_frames):
            if i not in choose_frames_id:
                continue
            else:
                good_sample = True
                for j in range(self.n_frames + 1):
                    if i+j not in _datas:
                        good_sample = False
                        break

                if good_sample:
                    _tmp = []
                    for j in range(self.n_frames + 1):
                        _tmp.append(_datas[i+j])
                    data_infos.append({
                        "data":_tmp,
                        "label":labels[i][0],
                        "hitter":labels[i][1]
                    })

        videos = []
        hitters = []

        for index in range(len(data_infos)):
            datas = data_infos[index]['data']
            hitter = data_infos[index]['hitter']
            hitters.append(hitter)
            video = []
            for data in datas:
                arr = cv2.imread(data['path'])
                assert arr is not None, "img path:{}".format(data['path'])
                bbox = data['bbox'][hitter]
                H, W = arr.shape[:2]
                x0 = int((bbox[0] - 0.5 *bbox[2]) * W)
                y0 = int((bbox[1] - 0.5 *bbox[3]) * H)
                x1 = int((bbox[0] + 0.5 *bbox[2]) * W)
                y1 = int((bbox[1] + 0.5 *bbox[3]) * H)

                arr = arr[y0:y1, x0:x1, :]
                t = torch.from_numpy(arr)
                t = t.float()
                t = t / 255.0
                t = t.permute(2, 0, 1)
                t = self.resize(t)
                video.append(t)
            video = torch.stack(video) # T, C, H, W

            video = self.transforms(video)
            video = video.permute(1, 0, 2, 3) 

            videos.append(video)
        
        return videos, hitters



if __name__ == "__main__":

    args = get_args()
    Path(args.save_root).mkdir(parents=True, exist_ok=True)

    modelA = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=True)
    modelA.blocks[5].pool.pool = torch.nn.AvgPool3d(kernel_size=(6, 6, 3), stride=1, padding=0)
    modelA.blocks[5].proj = torch.nn.Linear(in_features=args.in_features, out_features=args.out_features, bias=True)
    ckpt = torch.load(args.ckpt_A)
    modelA.load_state_dict(ckpt['model'])
    modelA.eval()
    modelA.cuda()

    modelB = torch.hub.load('facebookresearch/pytorchvideo', args.model_name, pretrained=True)
    modelB.blocks[5].pool.pool = torch.nn.AvgPool3d(kernel_size=(6, 6, 3), stride=1, padding=0)
    modelB.blocks[5].proj = torch.nn.Linear(in_features=args.in_features, out_features=args.out_features, bias=True)
    ckpt = torch.load(args.ckpt_B)
    modelB.load_state_dict(ckpt['model'])
    modelB.eval()
    modelB.cuda()

    loader = \
        DataLoader(entri_frames_root = args.entri_frames_root, 
                   yolo_pred_root = args.yolo_pred_root,
                   gt_root = args.gt_root,
                   data_size = args.data_size,
                   n_frames =  args.n_frames)

    pbar = tqdm.tqdm(total=1, ascii=True)
    for i in range(6,7):

        video_id = str(i).zfill(5)

        videos, hitters = loader.loadVideo(video_id)

        msg = ""
        for i in range(len(videos)):
            video = videos[i]
            video = video.cuda()
            video = video.unsqueeze(0)
            
            if hitters[i] == 'A':
                model = modelA
            else:
                model = modelB

            logit = model(video)[0]
            pred = int(torch.max(logit, dim=0)[1]) + 1
            
            msg += "{}\n".format(pred)
        #print(msg)
        
        with open(args.save_root + video_id + ".txt", "w") as ftxt:
            ftxt.write(msg)

        pbar.update(1)

    pbar.close()



