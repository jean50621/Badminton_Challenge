import torch
import torchvision.transforms as transforms
import torch.nn as nn
import csv
import os
import cv2
import tqdm
from copy import deepcopy
import random
from cfg import get_args

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
        self.ball_root = ball_root
        self.gt_root = gt_root
        self.empty_value = -100.0
        self.max_noise_number = 0.08
        self.x_shifts = [-30, 30]
        self.y_shifts = [-30, 30]
        self.data_size = data_size

        self.max_five_areas_length = 6

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

        for i in range(1, len(rows)):
            row = rows[i]
            
            if i == 1:
                start_frame = 0
            else:
                start_frame = int(rows[i-1][1])
            
            hitter = row[2]
            end_frame = int(row[1])

            label = int(row[3]) - 1

            areas.append([[start_frame, end_frame], label, hitter])

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
        video_name: val_00001, val_00002...
        """
        # === 1. load gt label
        csv_path = self.gt_root + "/" + folder + ".csv"
        areas = self.extract_csv_area(csv_path)
        
        # === 2. load obj locs and ball locs
        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        ball_path = self.ball_root + "/" + "val_" + folder + ".txt"
        balls = self.read_ball_txt(ball_path)

        img_paths = [x for x in os.listdir(self.entri_frames_root + "val_" + folder) if x.split('.')[-1] == 'jpg']  
        img_paths = set(img_paths)

        data_infos = []
        # === 3.build data_info
        for area in areas:

            start_frame = area[0][0]
            end_frame = area[0][1]
            if end_frame - start_frame < self.max_five_areas_length:
                end_frame += self.max_five_areas_length

            _tmp = {'folder': folder, 'target_hitter':None, 'ball':[], 'A':[] ,'B':[], 'net':[], 'court':[], 'imgs':[]}
            _tmp['target_hitter'] = area[2]
            skip_ids = []
            for i in range(start_frame, end_frame):
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
            for i in range(start_frame, end_frame):
                if i in skip_ids:
                    continue
                if i not in balls:
                    _tmp['ball'].append([self.empty_value, self.empty_value])
                else:
                    _tmp['ball'].append([balls[i][0], balls[i][1]])


            assert len(_tmp['ball']) == len(_tmp['B'])

            data_infos.append({"data":_tmp, 
                               "label":area[1]})


        packages = []

        self.max_area_length = 132 # dump code
        
        for index in range(len(data_infos)):

            package = {}

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

            target_hitter = data['target_hitter']

            video = []
            for i in video_selected_ids:
                arr = cv2.imread(data['imgs'][i])
                bbox = data[target_hitter][i]
                x, y, w, h = bbox
                xmin, xmax = x - 0.6 * w, x + 0.6 * w
                ymin, ymax = y - 0.6 * h, y + 0.6 * h
                xmin, xmax = int(xmin), int(xmax)
                ymin, ymax = int(ymin), int(ymax)
                xmin, xmax = max(xmin, 0), min(xmax, 1280)
                ymin, ymax = max(ymin, 0), min(ymax, 720) 
                t = deepcopy(arr[ymin:ymax, xmin:xmax, :])
                t = self.np2tensor(t)
                video.append(t)

            if len(video) != 0:
                _tmp = torch.stack(video)
                _tmp = self.transforms(_tmp)
                if _tmp.size(0) != self.max_five_areas_length:
                    empty_v = torch.zeros([self.max_five_areas_length, 3, self.data_size[0], self.data_size[1]]) - self.empty_value
                    empty_v[-1 * _tmp.size(0):] = _tmp
                    _tmp = empty_v
            else:
                _tmp = torch.zeros([self.max_five_areas_length, 3, self.data_size[0], self.data_size[1]]) - self.empty_value
            
            video = _tmp.permute(1, 0, 2, 3) # T, C, H, W -> C, T, H, W

            package['video'] = video
            
            ### 0. initial tensors
            ball = torch.zeros([self.max_area_length, 2]) + self.empty_value
            A = torch.zeros([self.max_area_length, 4]) + self.empty_value
            B = torch.zeros([self.max_area_length, 4]) + self.empty_value
            net = torch.zeros([self.max_area_length, 4]) + self.empty_value
            court = torch.zeros([self.max_area_length, 4]) + self.empty_value

            if data_length > 0:
                t_shift = random.randint(0, self.max_area_length - data_length)
                ### 1. build ball tensor
                _t = torch.tensor(data['ball'], dtype=torch.float32)
                ball[t_shift:t_shift+data_length] = _t
            package['ball'] = ball

            ### 2. build player A, player B, net, court tensors
            if data_length > 0:
                _t = torch.tensor(data['A'], dtype=torch.float32)
                A[t_shift:t_shift+data_length] = _t
            package['A'] = A
            if data_length > 0:
                _t = torch.tensor(data['B'], dtype=torch.float32)
                B[t_shift:t_shift+data_length] = _t
            package['B'] = B

            if data_length > 0:
                _t = torch.tensor(data['net'], dtype=torch.float32)
                net[t_shift:t_shift+data_length] = _t
            package['net'] = net

            if data_length > 0:
                _t = torch.tensor(data['court'], dtype=torch.float32)
                court[t_shift:t_shift+data_length] = _t
            package['court'] = court


            packages.append(package)
            
        return packages


if __name__ == "__main__":

    args = get_args()
    
    model = Model()
    ckpt = torch.load(args.weight)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    os.makedirs(args.save_root,exist_ok=True)
    loader = \
        DataLoader(args.entire_frame_root,
                   args.obj_root, 
                   args.ball_root,
                   args.gt_root,
                   args.data_size)

    pbar = tqdm.tqdm(total=1, ascii=True)
    for i in range(6, 7):

        video_id = str(i).zfill(5)

        packages = loader.loadVideo(video_id)

        msg = ""
        for i in range(len(packages)):
            with torch.no_grad():
                package = packages[i]
                ball, A, B, net, court = \
                    package['ball'], package['A'], package['B'], package['net'], package['court']
                ball, A, B, net, court = ball.cuda(), A.cuda(), B.cuda(), net.cuda(), court.cuda()
                ball, A, B, net, court = ball.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), net.unsqueeze(0), court.unsqueeze(0)
                video = package['video'].cuda()
                video = video.unsqueeze(0)
                score = None
                for _ in range(10):
                    out = model(video, ball, A, B, net, court)
                    if score is None:
                        score = torch.softmax(out[0], dim=0)
                    else:
                        score += torch.softmax(out[0], dim=0)
                
                score /= 10
                pred = int(torch.max(score, dim=0)[1]) + 1
                score = float(torch.max(score, dim=0)[0])
                msg += "{} {}\n".format(pred, score)
        
        with open(args.save_root + video_id + ".txt", "w") as ftxt:
            ftxt.write(msg)

        pbar.update(1)

    pbar.close()