import torch
import torchvision.transforms as transforms
import torch.nn as nn
import csv
import os
import cv2
import tqdm
import random
from cfg import get_args
from model import BT_Transformer

class DataLoader(object):

    def __init__(self, 
                 obj_root: str, 
                 ball_root: str,
                 gt_root: str):
        self.obj_root = obj_root
        self.ball_root = ball_root
        self.gt_root = gt_root
        self.empty_value = -100.0
        self.max_area_length = 168

    def extract_csv_area(self, csv_path):
        rows = []
        with open(csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)

        areas = []
        for i in range(len(rows[1:])):
            row = rows[i + 1]
            clip_label = -1
            start_frame = int(row[1])
            if i + 2 < len(rows):
                next_row = rows[i + 2]
                end_frame = int(next_row[1])
            else:
                end_frame = -1
            areas.append([[start_frame, end_frame], clip_label])

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

    def loadVideo(self, folder: str):

        csv_path = self.gt_root + "/" + folder + ".csv"
        areas = self.extract_csv_area(csv_path)

        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        ball_path = self.ball_root + "/" + "val_" + folder + ".txt"
        balls = self.read_ball_txt(ball_path)

        data_infos = []
        for area in areas:
            _tmp = {'ball':[], 'A':[] ,'B':[], 'net':[], 'court':[]}
            skip_ids = []
            end_frame = area[0][1]
            if end_frame == -1:
                end_frame = n_frames
            for i in range(area[0][0], end_frame):
                if i not in objs:
                    skip_ids.append(i)
                    continue
                if len(objs[i]['player']) != 2:
                    skip_ids.append(i)
                    continue
                _tmp['A'].append(objs[i]['player']['A'])
                _tmp['B'].append(objs[i]['player']['B'])
                _tmp['net'].append(objs[i]['net'])
                _tmp['court'].append(objs[i]['court'])

            skip_ids = set(skip_ids)
            for i in range(area[0][0], end_frame):
                if i in skip_ids:
                    continue
                if i not in balls:
                    _tmp['ball'].append([self.empty_value, self.empty_value])
                else:
                    _tmp['ball'].append([balls[i][0], balls[i][1]])

            assert len(_tmp['ball']) == len(_tmp['B'])

            data_infos.append({"data":_tmp})

        assert len(data_infos) == len(areas)

        balls, As, Bs, nets, courts = [], [], [], [], []
        for index in range(len(data_infos)):
            data = data_infos[index]['data']
            data_length = len(data['ball'])
            ### 0. initial tensors
            ball = torch.zeros([self.max_area_length, 2]) + self.empty_value
            A = torch.zeros([self.max_area_length, 4]) + self.empty_value
            B = torch.zeros([self.max_area_length, 4]) + self.empty_value
            net = torch.zeros([self.max_area_length, 4]) + self.empty_value
            court = torch.zeros([self.max_area_length, 4]) + self.empty_value

            if data_length != 0:
                t_shift = random.randint(0, self.max_area_length - data_length)
                ### 1. build ball tensor
                _t = torch.tensor(data['ball'], dtype=torch.float32)
                ball[t_shift:t_shift+data_length] = _t
                ball = self.gaussian_noise(data_length, t_shift, ball, 0.3)
            balls.append(ball)

            if data_length != 0:
                ### 2. build player A, player B, net, court tensors
                _t = torch.tensor(data['A'], dtype=torch.float32)
                A[t_shift:t_shift+data_length] = _t
                A = self.gaussian_noise(data_length, t_shift, A, 0.2)
            As.append(A)
            
            if data_length != 0:
                _t = torch.tensor(data['B'], dtype=torch.float32)
                B[t_shift:t_shift+data_length] = _t
                B = self.gaussian_noise(data_length, t_shift, B, 0.2)
            Bs.append(B)

            if data_length != 0:
                _t = torch.tensor(data['net'], dtype=torch.float32)
                net[t_shift:t_shift+data_length] = _t
                net = self.gaussian_noise(data_length, t_shift, net, 0.2)
            nets.append(net)

            if data_length != 0:
                _t = torch.tensor(data['court'], dtype=torch.float32)
                court[t_shift:t_shift+data_length] = _t
                court = self.gaussian_noise(data_length, t_shift, court, 0.2)
            courts.append(court)

        return balls, As, Bs, nets, courts


if __name__ == "__main__":

    args = get_args()
    

    model = BT_Transformer(args.num_classes,
                           args.length,
                           args.d_model,
                           args.dropout,
                           args.num_layers,
                           args.nhead,
                           args.dim_feedforward,
                           args.ffn_dropout,
                           args.att_dropout,
                           args.proj_dropout)
    ckpt = torch.load(args.weight )
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.cuda()
    
    loader = \
        DataLoader(args.obj_root, 
                   args.ball_root,
                   args.gt_root)
    os.makedirs(args.save_root,exist_ok=True)
    pbar = tqdm.tqdm(total=1, ascii=True)
    for i in range(6, 7):

        video_id = str(i).zfill(5)
        
        preds = {}
        n_areas = None
        n_test = 30
        for _ in range(n_test):

            balls, As, Bs, nets, courts = loader.loadVideo(video_id)
            n_areas = len(balls)
            for j in range(len(balls)):
                with torch.no_grad():
                    ball, A, B, net, court = balls[j], As[j], Bs[j], nets[j], courts[j]
                    ball, A, B, net, court = \
                        ball.cuda(), A.cuda(), B.cuda(), net.cuda(), court.cuda()
                    ball, A, B, net, court = \
                        ball.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0), net.unsqueeze(0), court.unsqueeze(0)
                    logit = model(ball, A, B, net, court)[0]

                    logit = logit.cpu()
                    logit = torch.softmax(logit, dim=0)
                    if j not in preds:
                        preds[j] = logit / n_test
                    else:
                        preds[j] += logit / n_test

        msg = ""
        for j in range(n_areas):
            pred = int(torch.max(preds[j], dim=0)[1]) + 1
            msg += "{}\n".format(pred)
        

        with open(args.save_root + video_id + ".txt", "w") as ftxt:
            ftxt.write(msg)

        pbar.update(1)

    pbar.close()

