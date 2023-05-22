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
                    _s = -1.0
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
            if i + 2 < len(rows):
                next_row = rows[i + 2]
                hit_frame = int(next_row[1])
                hitter = next_row[2]
                label_landingX = float(row[6])
                label_landingY = float(row[7])
                hits.append([hit_frame, hitter, label_landingX, label_landingY])

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
        ball_path = self.ball_root + "/" + "val_" + folder + ".txt" #val or test
        balls = self.read_ball_txt(ball_path)

        csv_path = self.gt_root + "/" + folder + ".csv"
        hits = self.extract_csv_hit_point(csv_path)
        print("hits", hits)
        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        img_paths = [x for x in os.listdir(self.entri_frames_root + "val_" + folder) if x.split('.')[-1] == 'jpg']  
        img_paths = set(img_paths)

        data_infos = []
        for hit in hits:
            hit_frame = hit[0]
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

            if _hit_frame != hit_frame:
                print(_hit_frame == hit_frame)
                print(_hit_frame, hit_frame)

            hit_frame = _hit_frame
            
            hitter = hit[1]
            label_landingX, label_landingY = hit[2:]
            for i in range(hit_frame - 1, hit_frame + 1): # return 2 results
                img_name = str(i) + ".jpg"
                if img_name not in img_paths:
                    continue
                data_infos.append({
                        "hit_frame":hit_frame,
                        "path":self.entri_frames_root + "val_" + folder + "/" + img_name,
                        "bbox":objs[i]['player'][hitter],
                        "label_landingX":label_landingX,
                        "label_landingY":label_landingY
                    })

        imgs = []
        xywhs = []
        hit_frames = []
        for index in range(len(data_infos)):
            arr = cv2.imread(data_infos[index]["path"])

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

            imgs.append(img)
            hit_frames.append(data_infos[index]["hit_frame"])
        print(hit_frames)

        return imgs, xywhs, hit_frames, balls


if __name__ == "__main__":

    """
    x range:  [-237, 243] 481
    y range:  [-33, 448] 482
    """
    args = get_args()
    model = Model()
    ckpt = torch.load(args.weight)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()

    loader = DataLoader(args.entri_frames_root,
                 args.obj_root, 
                 args.ball_root,
                 args.gt_root,
                 args.data_size)


    for i in range(6, 7):
        with torch.no_grad():
            idx = str(i).zfill(5)
            msg = ""
            imgs, xywhs, hit_frames, balls = loader.loadVideo(idx)
            for j in range(0, len(imgs), 2):
                p2x, p2y = 0, 0
                pred2_x, pred2_y = 0, 0
                for k in range(2):
                    img = imgs[j+1]
                    img = img.unsqueeze(0).cuda()
                    dx, dy = model(img)
                    # dx = torch.max(dx[0], dim=0)[1]
                    # dy = torch.max(dy[0], dim=0)[1]
                    dx = float(dx)
                    dy = float(dy)

                    
                    px, py = xywhs[j+1][0], xywhs[j+1][1]
                    pred_x, pred_y = px + dx, py + dy

                    p2x += px
                    p2y += py
                    pred2_x += pred_x
                    pred2_y += pred_y

                p2x /= 2
                p2y /= 2
                pred2_x /= 2
                pred2_y /= 2
                hit_f = hit_frames[j]
                hit_f -= 1

                # 前後追朔4 frames
                ball_locs = []
                ball_frames = []
                for hi in range(hit_f-4, hit_f+4):
                    if hi in balls:
                        ball_locs.append(balls[hi])
                        ball_frames.append(hi)

                vecs = []
                for hi in range(len(ball_locs)-1):
                    v = [ball_locs[hi+1][0] - ball_locs[hi][0], ball_locs[hi+1][1] - ball_locs[hi][1]]
                    vecs.append(v)

                min_sim, min_idx = None, None
                for vi in range(len(vecs)-1):
                    norm1 = (vecs[vi + 1][0] ** 2 + vecs[vi + 1][1] ** 2) ** 0.5
                    norm2 = (vecs[vi][0] ** 2 + vecs[vi][1] ** 2) ** 0.5
                    sim = (vecs[vi + 1][0] * vecs[vi][0] + vecs[vi + 1][1] * vecs[vi][1]) / (norm1 * norm2 + 1)
                    if min_sim is None or sim < min_sim:
                        min_sim = sim
                        min_idx = vi

                ball_x = None
                if min_idx is not None:

                    _hit_f = int(round((ball_frames[min_idx] + ball_frames[min_idx+1]) / 2))
                    if _hit_f in balls:
                        ball_x = balls[_hit_f][0]

                if ball_x is None:
                    hfffff = hit_f
                    if hit_f not in balls:
                        prvs_i = loader.find_prvs(balls, hit_f, hit_f-2)
                        next_i = loader.find_next(balls, hit_f, hit_f+2)
                        if prvs_i is not None and next_i is not None:
                            if abs(prvs_i - hit_f) <= abs(next_i - hit_f):
                                hfffff = prvs_i
                            else:
                                hfffff = next_i
                        elif prvs_i is not None:
                            hfffff = prvs_i
                        elif next_i is not None:
                            hfffff = prvs_i
                        else:
                            hfffff = None

                    if hfffff is not None:
                        ball_x = balls[hfffff][0]

                if ball_x is not None:
                    if abs(ball_x - pred2_x) <= 20:
                        pred2_x = (ball_x + pred2_x) / 2
                    elif 20 < abs(ball_x - pred2_x) <= 25:
                        pred2_x = (ball_x + pred2_x * 2) / 3

                msg += "{} {} {} {} {} {}\n".format(hit_f, p2x, p2y, pred2_x, pred2_y, ball_x)

        os.makedirs(args.save_root,exist_ok=True)
        with open(args.save_root +"{}.txt".format(idx), "w") as ftxt:
            ftxt.write(msg)


    

