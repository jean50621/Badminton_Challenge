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
        
        print("start building dataset...")

        self.entri_frames_root = entri_frames_root
        self.obj_root = obj_root
        self.ball_root = ball_root
        self.gt_root = gt_root
        self.empty_value = -100.0
        self.max_noise_number = 0.08
        self.x_shifts = [-30, 30]
        self.y_shifts = [-30, 30]
        self.data_size = data_size

        self.max_five_areas_length = 32
        self.resize = transforms.Resize((data_size[0], data_size[1]))

        self.transforms = transforms.Compose([
            transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        ])

    def extract_csv_area(self, csv_path):
        rows = []
        with open(csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)

        areas = []
        row = rows[len(rows) - 1]
        start_frame = int(row[1])
        end_frame = -1
        if row[13] == 'A':
            winner_label = 0
        else:
            winner_label = 1

        if row[2] == 'A':
            last_hitter = [1, 0]
        else:
            last_hitter = [0, 1]
        areas.append([[start_frame, end_frame], winner_label, last_hitter])

        return areas


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
            else:
                if objs[i - 1]['player']['A'] is None and objs[i - 1]['player']['B'] is None:
                    del objs[i - 1]
                elif objs[i - 1]['player']['A'] is not None and objs[i - 1]['player']['B'] is None:
                    objs[i - 1]['player']['B'] = objs[i - 1]['player']['A']

        return objs

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

    def find_prvs(self, infos: dict, i: int):
        for j in range(i-1, -1, -1):
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




    def gaussian_noise(self, data_length, t_shifts, data, scale=10.0):
        noise_number = random.randint(0, data_length)
        if noise_number != 0:
            noise_ids = torch.randperm(data_length)[:noise_number] + t_shifts
            n = torch.randn_like(data[noise_ids]) * scale
            data[noise_ids] += n
        return data

    def np2tensor(self, arr):
        t = torch.from_numpy(arr)
        t = t.float()
        t = t / 255.0
        t = t.permute(2, 0, 1)
        t = self.resize(t)
        return t

    
    def loadVideo(self, folder: str):
        """
        folder: 00001, 00002, ...
        """
        # === 1. load gt label
        csv_path = self.gt_root + "/" + folder + ".csv"
        areas = self.extract_csv_area(csv_path)
        
        # === 2. load obj locs and ball locs
        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        if not os.path.isdir(obj_path):
            print(obj_path)
            return
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        ball_path = self.ball_root + "/" + "val_" + folder + ".txt"
        balls = self.read_ball_txt(ball_path)

        img_paths = [x for x in os.listdir(self.entri_frames_root + "val_" + folder) if x.split('.')[-1] == 'jpg']  
        img_paths = set(img_paths)

        data_infos = []
        for area in areas:
            _tmp = {'folder':folder, 'last_hitter':None, 'ball':[], 'A':[] ,'B':[], 'net':[], 'court':[], 'imgs':[]}
            _tmp['last_hitter'] = area[2]
            skip_ids = []
            end_frame = area[0][1]
            if end_frame == -1:
                end_frame = n_frames
            for i in range(area[0][0], end_frame):
                if i not in objs:
                    skip_ids.append(i)
                    continue
                img_name = "{}.jpg".format(i)
                if img_name not in img_paths:
                    raise ValueError("Why on earth there is not {i}.jpg in the img_paths???")
                if len(objs[i]['player']) != 2:
                    skip_ids.append(i)
                    continue
                _tmp['A'].append(objs[i]['player']['A'])
                _tmp['B'].append(objs[i]['player']['B'])
                _tmp['net'].append(objs[i]['net'])
                _tmp['court'].append(objs[i]['court'])
                _tmp['imgs'].append(self.entri_frames_root + "val_" + folder + "/" + img_name)

            skip_ids = set(skip_ids)
            for i in range(area[0][0], end_frame):
                if i in skip_ids:
                    continue
                if i not in balls:
                    _tmp['ball'].append([self.empty_value, self.empty_value])
                else:
                    _tmp['ball'].append([balls[i][0], balls[i][1]])


            assert len(_tmp['ball']) == len(_tmp['B'])

            data_infos.append({"data":_tmp, 
                               "label":area[1]})


        assert len(data_infos) == 1
        index = 0

        package = {}

        self.max_area_length = 168 # dump code
        
        data = data_infos[index]['data']

        data_length = len(data['ball'])
        ### five video stack
        video_selected_ids = []
        if data_length < self.max_five_areas_length:
            for i in range(data_length):
                video_selected_ids.append(i)
        else:
            for i in range(data_length - self.max_five_areas_length, data_length):
                video_selected_ids.append(i)

        video = {'up':[], 'right':[], 'down':[], 'left':[], 'net':[]}
        border_size = 0.28
        for i in video_selected_ids:
            arr = cv2.imread(data['imgs'][i])
            court_bbox = data['court'][i]
            xmin, xmax = court_bbox[0] - 0.5 * court_bbox[2], court_bbox[0] + 0.5 * court_bbox[2]
            ymin, ymax = court_bbox[1] - 0.5 * court_bbox[3], court_bbox[1] + 0.5 * court_bbox[3]
            ## up
            x0, x1 = max(xmin, 0), min(xmax, 1280)
            y0, y1 = max(ymin - border_size * 0.5 * court_bbox[3], 0), min(ymin + border_size * court_bbox[3], 1280)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            arr1 = deepcopy(arr[y0:y1, x0:x1, :])
            t = self.np2tensor(arr1)
            video['up'].append(t)
            ## right
            x0, x1 = max(xmax - border_size * court_bbox[2], 0), min(xmax + border_size * 0.5 * court_bbox[2], 1280)
            y0, y1 = max(ymin, 0), min(ymax, 720)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            arr2 = deepcopy(arr[y0:y1, x0:x1, :])
            t = self.np2tensor(arr2)
            video['right'].append(t)
            ## down
            x0, x1 = max(xmin, 0), min(xmax, 1280)
            y0, y1 = max(ymax - border_size * court_bbox[3], 0), min(ymax + border_size * 0.5 * court_bbox[3], 1280)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            arr3 = deepcopy(arr[y0:y1, x0:x1, :])
            t = self.np2tensor(arr3)
            video['down'].append(t)
            ## left
            x0, x1 = max(xmin - border_size * 0.5 * court_bbox[2], 0), min(xmin + border_size * court_bbox[2], 1280)
            y0, y1 = max(ymin, 0), min(ymax, 720)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            arr4 = deepcopy(arr[y0:y1, x0:x1, :])
            t = self.np2tensor(arr4)
            video['left'].append(t)
            ## net
            net_bbox = data['net'][i]
            xmin, xmax = net_bbox[0] - 0.5 * net_bbox[2], net_bbox[0] + 0.5 * net_bbox[2]
            ymin, ymax = net_bbox[1], net_bbox[1] + 1.0 * net_bbox[3]
            x0, x1 = max(xmin, 0), min(xmax, 1280)
            y0, y1 = max(ymin, 0), min(ymax, 720)
            x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
            arr5 = deepcopy(arr[y0:y1, x0:x1, :])
            t = self.np2tensor(arr5)
            video['net'].append(t)

        videos = []
        for name in video:
            _tmp = torch.stack(video[name])
            _tmp = self.transforms(_tmp)
            if _tmp.size(0) != self.max_five_areas_length:
                empty_v = torch.zeros([self.max_five_areas_length, 3, self.data_size[0], self.data_size[1]]) - self.empty_value
                empty_v[-1 * _tmp.size(0):] = _tmp
                _tmp = empty_v
            # video[name] = _tmp
            videos.append(_tmp)
        videos = torch.stack(videos, dim=0) # N, T, C, H, W
        videos = videos.permute(0, 2, 1, 3, 4) # N, C, T, H, W

        package['videos'] = videos
        package['last_hitter'] = torch.tensor(data['last_hitter'], dtype=torch.float32)

        ### 0. initial tensors
        ball = torch.zeros([self.max_area_length, 2]) + self.empty_value
        A = torch.zeros([self.max_area_length, 4]) + self.empty_value
        B = torch.zeros([self.max_area_length, 4]) + self.empty_value
        net = torch.zeros([self.max_area_length, 4]) + self.empty_value
        court = torch.zeros([self.max_area_length, 4]) + self.empty_value

        t_shift = 0
        ### 1. build ball tensor
        _t = torch.tensor(data['ball'], dtype=torch.float32)
        ball[t_shift:t_shift+data_length] = _t
        package['ball'] = ball

        ### 2. build player A, player B, net, court tensors
        _t = torch.tensor(data['A'], dtype=torch.float32)
        A[t_shift:t_shift+data_length] = _t
        package['A'] = A
        _t = torch.tensor(data['B'], dtype=torch.float32)
        B[t_shift:t_shift+data_length] = _t
        package['B'] = B

        _t = torch.tensor(data['net'], dtype=torch.float32)
        net[t_shift:t_shift+data_length] = _t
        package['net'] = net

        _t = torch.tensor(data['court'], dtype=torch.float32)
        court[t_shift:t_shift+data_length] = _t
        package['court'] = court

        return package




if __name__ == "__main__":


    args = get_args()
    model = Model()
    ckpt = torch.load(args.weight)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    os.makedirs(args.save_root,exist_ok=True)

    loader = DataLoader(args.entri_frames_root,
                 args.obj_root, 
                 args.ball_root,
                 args.gt_root,
                 args.data_size)


    for i in range(6, 7):
        idx = str(i).zfill(5)
        print(idx)
        pred = None
        n_test = 30
        for _ in range(n_test):
            with torch.no_grad():
                package = loader.loadVideo(idx)
                ball, A, B, net, court = \
                    package['ball'], package['A'], package['B'], package['net'], package['court']
                ball, A, B, net, court = ball.cuda(), A.cuda(), B.cuda(), net.cuda(), court.cuda()
                ball, A, B, net, court = ball.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), net.unsqueeze(0), court.unsqueeze(0)
                videos = package['videos'].cuda()
                videos = videos.unsqueeze(0)
                last_hitter = package['last_hitter'].cuda()
                last_hitter = last_hitter.unsqueeze(0)

                logit = model(videos, last_hitter, ball, A, B, net, court)[0]

                logit = logit.cpu()
                logit = torch.softmax(logit, dim=0)
                if pred is None:
                    pred = logit / n_test
                else:
                    pred += logit / n_test
        
        pred = int(torch.max(pred, dim=0)[1])
        if pred == 0:
            pred = 'A'
        else:
            pred = 'B'

        msg = "{}".format(pred)
        print("    ", msg)
        
        
        with open(args.save_root + "{}.txt".format(idx), "w") as ftxt:
            ftxt.write(msg)


    

