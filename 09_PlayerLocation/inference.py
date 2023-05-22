import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import csv
from PIL import Image
from copy import deepcopy
from cfg import get_args
import math
from tqdm import tqdm
import os

from model import Model

class DataLoader(object):

    def __init__(self,
                 entri_frames_root: str,
                 obj_root: str, 
                 ball_root: str,
                 gt_root: str,
                 data_size : list):
        self.entri_frames_root = entri_frames_root
        self.obj_root = obj_root
        self.gt_root = gt_root
        self.data_size = data_size

        self.ball_root = ball_root

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((data_size[0], data_size[1])),
            transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        ])

    def load_balls(self, path):
        with open(path, "r") as ftxt:
            datas = ftxt.read().split('\n')
        datas = [x for x in datas if x != '']
        infos = {}
        max_frames = None
        for data in datas:
            tmp = data.split(' ')
            frame_id = int(tmp[0]) + 4
            x = float(tmp[1])
            y = float(tmp[2])
            score = float(tmp[3])
            good_det = int(tmp[4]) == 1
            if good_det:
                infos[frame_id] = [x, y, score]
        return infos

    def read_ball_txt(self, path: str) -> [dict, int]:
        with open(path, "r") as ftxt:
            datas = ftxt.read().split('\n')
        datas = [x for x in datas if x != '']
        infos = {}
        ball_loc = []
        max_frames = None
        total_hit = None
        for data in datas:
            tmp = data.split(' ')
            frame_id = int(tmp[0]) + 4
            x = float(tmp[1])
            y = float(tmp[2])
            score = float(tmp[3])
            good_det = int(tmp[4]) == 1
            if max_frames is None or frame_id > max_frames:
                max_frames = frame_id
            if good_det:
                infos[frame_id] = [x, y, score]

        vecs = self.calculate_vecs(infos, max_frames)
        infos = self.interplote(infos, vecs, max_frames)
        vecs = self.calculate_vecs(infos, max_frames)
        noise_frames = self.filter_out_first_noise(vecs, max_frames)
        if len(noise_frames) > 0 and noise_frames[0] < 60: # first two secs
            # remove all results before noise_frames[0]
            for i in range(noise_frames[0], -1, -1):
                if i in infos:
                    del infos[i]
        return infos

    def find_prvs(self, infos: dict, i: int, max_retrive: int = None):
        if max_retrive is None:
            final = -1
        else:
            final = max_retrive
        for j in range(i-1, final, -1):
            if j in infos:
                return j
        return None

    def find_next(self, infos: dict, i: int, max_frames: int):
        for j in range(i+1, max_frames+1):
            if j in infos:
                return j
        return None


    def fill_value(self, prvs_i: int, next_i: int, target_i: int, prvs_val: float, next_val: float) -> float:
        if (next_val - prvs_val) > 200:
            return None
        gap = next_i - prvs_i
        prop = (target_i - prvs_i) / gap
        diff = next_val - prvs_val
        target_val = prvs_val + diff * prop
        return target_val

    def calculate_vecs(self, infos: dict, max_frames: int):
        vecs = {}
        for i in range(max_frames + 1):
            if i in infos and i-1 in infos:
                _v = [infos[i][0] - infos[i-1][0], infos[i][1] - infos[i-1][1]]
                j = 2
                while i - j > 0:
                    if i - j not in infos:
                        j += 1
                        continue
                    if _v[0] == 0 and _v[1] == 0:
                        _v = [infos[i][0] - infos[i-j][0], infos[i][1] - infos[i-j][1]]
                    else:
                        break
                    j += 1
                vecs[i] = _v
        return vecs

    def filter_out_first_noise(self, vecs: dict, max_frames: int):
        noise_frames = []
        for i in range(max_frames):
            if i in vecs:
                if abs(vecs[i][0]) > 80 or abs(vecs[i][1]) > 100:
                    noise_frames.append(i)
        return noise_frames


    def interplote(self, infos: dict, vecs: dict, max_frames: int) -> dict:
        for i in range(max_frames + 1):
            if i not in infos:
                prvs_i = self.find_prvs(infos, i)
                next_i = self.find_next(infos, i, max_frames)
                if prvs_i is not None and next_i is not None:
                    if prvs_i in vecs:
                        if infos[prvs_i][1] + vecs[prvs_i][1] < 0:
                            continue
                    if abs(infos[prvs_i][0] - infos[next_i][0]) > 150 or abs(infos[prvs_i][1] - infos[next_i][1]) > 200:
                        continue
                    _x = self.fill_value(prvs_i, next_i, i, infos[prvs_i][0], infos[next_i][0])
                    _y = self.fill_value(prvs_i, next_i, i, infos[prvs_i][1], infos[next_i][1])
                    _s = self.fill_value(prvs_i, next_i, i, infos[prvs_i][2], infos[next_i][2])
                    if _x is None or _y is None:
                        continue
                    infos[i] = [_x, _y, _s]

        return infos

    def extract_csv_hit_point(self, csv_path):
        rows = []
        with open(csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)

        hits = []
        for i in range(len(rows[1:])):
            row = rows[i + 1]
            hit_frame = int(row[1])
            hitter = row[2]
            hitterX = float(row[8])
            hitterY = float(row[9])
            otherhitterX = float(row[10]) 
            otherhitterY = float(row[11])
            hits.append([hit_frame, hitter, hitterX, hitterY, otherhitterX, otherhitterY])

        return hits


    def load_objs(self, root: str, n_frames: int):
        objs = {}
        cls_table = {
            0:'player',
            1:'net',
            2:'court'
        }
        for i in range(1, n_frames + 1):
            
            txt_path = root + "/{}.txt".format(i)
            
            if not os.path.isfile(txt_path):
                # print("        object bbox file not found: ", txt_path)
                continue
            
            with open(txt_path, "r") as ftxt:
                datas = ftxt.read().split('\n')
            
            objs[i - 1] = {'player':{'A':None, 'B':None}}
            for data in datas:
                if data == '':
                    continue
                infos = data.split(' ')
                obj_id = int(infos[0])
                img_w, img_h = 1280, 720
                x = float(infos[1]) * img_w
                y = float(infos[2]) * img_h
                w = float(infos[3]) * img_w
                h = float(infos[4]) * img_h
                cls_name = cls_table[obj_id]
                if cls_name == 'player':
                    if objs[i - 1]['player']['A'] is None:
                        objs[i - 1]['player']['A'] = [x, y, w, h]
                    else:
                        need_swap = False
                        if y < objs[i - 1]['player']['A'][1]:
                            need_swap = True
                        if need_swap:
                            tmp = objs[i - 1]['player']['A']
                            objs[i - 1]['player']['A'] = [x, y, w, h]
                            objs[i - 1]['player']['B'] = tmp
                        else:
                            objs[i - 1]['player']['B'] = [x, y, w, h]
                else:
                    objs[i - 1][cls_name] = [x, y, w, h]
            if 'net' not in objs[i - 1]:
                del objs[i - 1]

        return objs


    
    def loadVideo(self, folder: str):
        """
        folder: 00001, 00002, ...
        """
        ball_path = self.ball_root + "/" + "val_" + folder + ".txt"
        balls = self.read_ball_txt(ball_path)

        csv_path = self.gt_root + "/" + folder + ".csv"
        hits = self.extract_csv_hit_point(csv_path)

        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        img_paths = [x for x in os.listdir(self.entri_frames_root + "val_" + folder) if x.split('.')[-1] == 'jpg']  
        img_paths = set(img_paths)

        data_infos = []
        for hit in hits:
            hit_frame = hit[0]
            hitter = hit[1]
            if hitter == 'A':
                other_hitter = 'B'
            else:
                other_hitter = 'A'
            hitterX, hitterY, otherhitterX, otherhitterY = hit[2:]
            for i in range(hit_frame - 1, hit_frame + 1):
                img_name = str(i) + ".jpg"
                if img_name not in img_paths:
                    print("???")
                    continue

                if i not in objs:
                    _hit_frame = hit_frame
                    min_dis = 10000
                    min_frame = None
                    if hit_frame not in objs:
                        for j in range(-50, 1, 1):
                            if hit_frame + j in objs:
                                _hit_frame = hit_frame + j
                                if abs(_hit_frame - hit_frame) < min_dis:
                                    min_dis = abs(_hit_frame - hit_frame) 
                                    min_frame = _hit_frame
                        _hit_frame = min_frame
                    # print(_hit_frame == hit_frame)
                    if _hit_frame != hit_frame:
                        print("[2]", _hit_frame, hit_frame)

                    jjj = _hit_frame
                else:
                    jjj = i

                data_infos.append({
                        "hitter":hitter,
                        "hit_frame":hit_frame,
                        "path":self.entri_frames_root + "val_" + folder + "/" + img_name,
                        "bbox":objs[jjj]['player'][hitter],
                        "other_bbox":objs[jjj]['player'][other_hitter],
                        "hitterX":hitterX,
                        "hitterY":hitterY,
                        "otherhitterX":otherhitterX,
                        "otherhitterY":otherhitterY
                    })

        imgs = []
        oimgs = []
        hitter_infos = []
        ohitter_infos = []
        xywhs = []
        oxywhs = []
        hit_frames = []
        hitters = []
        for index in range(len(data_infos)):
            arr = cv2.imread(data_infos[index]["path"])

            hitters.append(data_infos[index]['hitter'])

            px, py, pw, ph = data_infos[index]["bbox"]
            xywhs.append([px, py])

            x0 = max(int(px - 1.0 * pw), 0)
            x1 = min(int(px + 1.0 * pw), 1280)
            y0 = max(int(py - 0.62 * ph), 0)
            y1 = min(int(py + 0.78 * ph), 720)

            arr = arr[y0:y1, x0:x1, :]
            arr = arr[:, :, ::-1]
            img = Image.fromarray(arr)
            img = self.transforms(img)

            arr = cv2.imread(data_infos[index]["path"])
            r = data_infos[index]["other_bbox"]
            if r is not None:
                px, py, pw, ph = r
            oxywhs.append([px, py])

            x0 = max(int(px - 1 * pw), 0)
            x1 = min(int(px + 1 * pw), 1280)
            y0 = max(int(py - 10 - 0.62 * ph), 0)
            y1 = min(int(py - 10 + 0.78 * ph), 720)
            
            arr = arr[y0:y1, x0:x1, :]
            arr = arr[:, :, ::-1]
            oimg = Image.fromarray(arr)
            oimg = self.transforms(oimg)

            hitter_info = torch.tensor([1, 0], dtype=torch.float32)
            ohitter_info = torch.tensor([0, 1], dtype=torch.float32)

            hitter_infos.append(hitter_info)
            ohitter_infos.append(ohitter_info)

            oimgs.append(oimg)
            imgs.append(img)
            hit_frames.append(data_infos[index]["hit_frame"])

        return imgs, oimgs, xywhs, oxywhs, hit_frames, hitters, hitter_infos, ohitter_infos


def forward(model, data, info_data):
    x = model.model(data)
    x = model.pool(x)
    hs = model.hitter_info_proj(info_data)
    x = x.flatten(1)
    x = torch.cat([x, hs], dim=1)
    out_x = model.head1(x)
    out_y = model.head2(x)
    return out_x, out_y


if __name__ == "__main__":

    """
    x range:  [-237, 243] 481
    y range:  [-33, 448] 482
    """
    args = get_args()
    model = Model()
    model.head1 = nn.Sequential(
            nn.Linear(1792 + 128, 1792),
            # nn.ReLU(),
            nn.Linear(1792, 1) # 434
        )
    model.head2 = nn.Sequential(
            nn.Linear(1792 + 128, 1792),
            # nn.ReLU(),
            nn.Linear(1792, 1) # 483
        )
    model.hitter_info_proj = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
    ckpt = torch.load(args.weight)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()

    loader = DataLoader(args.entri_frames_root,
                 args.obj_root, 
                 args.ball_root,
                 args.gt_root,
                 args.data_size)

    for i in range(6,7):
        with torch.no_grad():
            idx = str(i).zfill(5)
            msg = ""
            imgs, oimgs, xywhs, oxywhs, hit_frames, hitters, hitter_infos, ohitter_infos = \
                loader.loadVideo(idx)
            for j in range(0, len(imgs), 2):
                p2x1, p2y1 = 0, 0
                pred2_x1, pred2_y1 = 0, 0

                p2x2, p2y2 = 0, 0
                pred2_x2, pred2_y2 = 0, 0

                for k in range(2):
                    img = imgs[j+k]
                    img = img.unsqueeze(0).cuda()
                    hitter_info = hitter_infos[j+k]
                    hitter_info = hitter_info.cuda().unsqueeze(0)
                    dx, dy = forward(model, img, hitter_info)
                    dx = float(dx)
                    dy = float(dy)
                    
                    px, py = xywhs[j+k][0], xywhs[j+k][1]
                    pred_x, pred_y = px + dx, py + dy

                    p2x1 += px
                    p2y1 += py
                    pred2_x1 += pred_x
                    pred2_y1 += pred_y

                    oimg = oimgs[j+k]
                    oimg = oimg.unsqueeze(0).cuda()
                    ohitter_info = ohitter_infos[j+k]
                    ohitter_info = ohitter_info.cuda().unsqueeze(0)
                    dx, dy = forward(model, oimg, ohitter_info)
                    dx = float(dx)
                    dy = float(dy)
                    
                    px, py = oxywhs[j+k][0], oxywhs[j+k][1]
                    pred_x, pred_y = px + dx, py + dy

                    p2x2 += px
                    p2y2 += py
                    pred2_x2 += pred_x
                    pred2_y2 += pred_y


                p2x1 /= 2
                p2y1 /= 2
                pred2_x1 /= 2
                pred2_y1 /= 2
                p2x2 /= 2
                p2y2 /= 2
                pred2_x2 /= 2
                pred2_y2 /= 2
                hit_f = hit_frames[j]
                
                msg += "{} {} {} {} {} {} {} {} {} {}\n".format(hit_f, hitters[j], p2x1, p2y1, pred2_x1, pred2_y1,
                    p2x2, p2y2, pred2_x2, pred2_y2)
        os.makedirs(args.save_root,exist_ok=True)
        with open(args.save_root+"{}.txt".format(idx), "w") as ftxt:
            ftxt.write(msg)


    

