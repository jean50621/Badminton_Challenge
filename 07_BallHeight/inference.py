import torch
import torchvision.transforms as transforms
import torch.nn as nn
import csv
import os
import cv2
import tqdm

from cfg import get_args
from model import Model

class DataLoader(object):

    def __init__(self, 
                 entire_frame_root: str,
                 obj_root: str, 
                 ball_root: str,
                 gt_root: str,
                 n_frames: int = 4,
                 data_size: list = [224, 224]):
        self.entire_frame_root = entire_frame_root
        self.obj_root = obj_root
        self.ball_root = ball_root
        self.gt_root = gt_root
        self.n_frames = n_frames

        self.data_size = data_size
        self.resize = transforms.Resize((data_size[0], data_size[1]))
        self.transforms = transforms.Compose([
            transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        ])

    def extract_csv_hit_point(self, csv_path):
        rows = []
        with open(csv_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)

        hit_points = []
        for i in range(len(rows[1:])):
            row = rows[i + 1]
            hit_frame = int(row[1])
            if i + 2 < len(rows):
                next_hit_frame = int(rows[i + 2][1])
            else:
                next_hit_frame = -1
            if i > 0:
                prvs_hit_frame = int(rows[i][1])
            else:
                prvs_hit_frame = 0

            hit_player = row[2]
            label_bHeight = int(row[5]) - 1
            label_landingX = int(row[6])
            label_landingY = int(row[7])
            
            hit_points.append([hit_frame, [prvs_hit_frame, next_hit_frame], \
                hit_player, label_bHeight, label_landingX, label_landingY])

        return hit_points

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

    def l2_distance(self, p1, p2):
        dis = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        return dis

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

    def loadVideo(self, folder: str):
        """
        video_name: val_00001, val_00002...
        """
        csv_path = self.gt_root + "/" + folder + ".csv"
        hit_points = self.extract_csv_hit_point(csv_path)

        obj_path = self.obj_root + "/" + "val_" + folder + "/"
        frame_numbers = [int(x.split('.')[0]) for x in os.listdir(obj_path)]
        n_frames = max(frame_numbers)
        objs = self.load_objs(obj_path, n_frames)

        ball_path = self.ball_root + "/" + "val_" + folder + ".txt"
        balls = self.load_balls(ball_path)

        img_paths = [x for x in os.listdir(self.entire_frame_root + "val_" + folder) if x.split('.')[-1] == 'jpg']  
        img_paths = set(img_paths)

        data_infos = []
        ### build data infos
        for hit_pt in hit_points:
            hit_frame = hit_pt[0]

            target_player = hit_pt[2]

            if hit_frame in balls:
                hit_x, hit_y = balls[hit_frame][:2]
            else:
                x_t1, y_t1 = None, None
                x_t2, y_t2 = None, None
                x_t3, y_t3 = None, None

                for i in range(hit_frame-1, -1, -1):
                    if i in balls:
                        x_t1, y_t1 = balls[i][:2]
                        break
                for i in range(hit_frame+1, n_frames, 1):
                    if i in balls:
                        x_t3, y_t3 = balls[i][:2]
                        break
                if x_t1 is not None and x_t3 is not None:
                    x_t2, y_t2 = (x_t1 + x_t3) / 2, (y_t1 + y_t3) / 2

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

                px = objs[_hit_frame]['player'][target_player][0]
                py = objs[_hit_frame]['player'][target_player][1] - 0.2 * objs[_hit_frame]['player'][target_player][3]
                d1, d2, d3  = 10 ** 5, 10 ** 5, 10 ** 5
                if x_t1 is not None:
                    d1 = self.l2_distance([x_t1, y_t1], [px, py])
                if x_t2 is not None:
                    d2 = self.l2_distance([x_t2, y_t2], [px, py])
                if x_t3 is not None:
                    d3 = self.l2_distance([x_t3, y_t3], [px, py])
                if d1 < d2 and d1 < d3:
                    hit_x, hit_y = x_t1, y_t1
                elif d3 < d1 and d3 < d2:
                    hit_x, hit_y = x_t3, y_t3
                elif d2 < d3 and d2 < d1:
                    hit_x, hit_y = x_t2, y_t2
                else:
                    hit_x, hit_y = px, py

            action_img_paths = []
            player_bboxs = []
            miss_count = 0
            for i in range(hit_frame - self.n_frames, hit_frame):
                if hit_frame not in img_paths:
                    miss_count += 1
                    continue
                if hit_frame not in objs:
                    miss_count += 1
                    continue
                if objs[hit_frame]['player'][target_player] is None:
                    miss_count += 1
                    continue

                img_name = "{}.jpg".format(i)
                action_img_paths.append(self.entire_frame_root + "val_" + folder + "/" + img_name)
                player_bboxs.append(objs[hit_frame]['player'][target_player])

            for i in range(hit_frame + 1 , hit_frame + self.n_frames + 1):
                if hit_frame not in img_paths:
                    miss_count += 1
                    continue
                if hit_frame not in objs:
                    miss_count += 1
                    continue
                if objs[hit_frame]['player'][target_player] is None:
                    miss_count += 1
                    continue
                img_name = "{}.jpg".format(i)
                action_img_paths.append(self.entire_frame_root + "val_" + folder + "/" + img_name)
                player_bboxs.append(objs[hit_frame]['player'][target_player])

            for i in range(miss_count + 1):
                img_name = "{}.jpg".format(hit_frame)
                action_img_paths.append(self.entire_frame_root + "val_" + folder + "/" + img_name)
                
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

                hit_frame = _hit_frame

                player_bboxs.append(objs[_hit_frame]['player'][target_player])

            ## data
            data_infos.append({
                    "paths":action_img_paths,
                    "bboxs":player_bboxs,
                    "net":objs[hit_frame]['net'],
                    "ball":[hit_x, hit_y],
                    "court":objs[hit_frame]['court'],
                    "player":objs[hit_frame]['player'][target_player],
                    "bHeight":hit_pt[3], # label
                })

        assert len(data_infos) == len(hit_points), "val_{}".format(folder)

        videos, nets, balls, courts, players, label_bHeights = [], [], [], [], [], []
        for index in range(len(data_infos)):
            img_paths = data_infos[index]['paths']
            img_bboxs = data_infos[index]['bboxs']

            video = []
            for i in range(len(img_paths)):
                arr = cv2.imread(img_paths[i])
                x, y, w, h = img_bboxs[i]
                imgH, imgW = arr.shape[:2]
                x0 = max(int((x - 0.68 * w)), 0)
                x1 = min(int((x + 0.68 * w)), imgW)
                y0 = max(int((y - 0.68 * h)), 0)
                y1 = min(int((y + 0.52 * h)), imgH)
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

            net = torch.tensor(data_infos[index]["net"], dtype=torch.float32)
            ball = torch.tensor(data_infos[index]["ball"], dtype=torch.float32)
            court = torch.tensor(data_infos[index]["court"], dtype=torch.float32)
            player = torch.tensor(data_infos[index]["player"], dtype=torch.float32)
            label_bHeight = data_infos[index]["bHeight"]

            videos.append(video)
            nets.append(net)
            balls.append(ball)
            courts.append(court)
            players.append(player)
            label_bHeights.append(label_bHeight)
            
        return videos, nets, balls, courts, players, label_bHeights


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
                   args.gt_root)

    pbar = tqdm.tqdm(total=1, ascii=True)
    for i in range(6, 7):

        video_id = str(i).zfill(5)

        videos, nets, balls, courts, players, label_bHeights = loader.loadVideo(video_id)

        msg = ""
        for i in range(len(videos)):
            video, net, ball, court, player = videos[i], nets[i], balls[i], courts[i], players[i]
            video, net, ball, court, player = \
                video.cuda(), net.cuda(), ball.cuda(), court.cuda(), player.cuda()
            video, net, ball, court, player = \
                video.unsqueeze(0), net.unsqueeze(0), ball.unsqueeze(0), court.unsqueeze(0), player.unsqueeze(0)
            logit = model(video, net, ball, court, player)[0]
            pred = int(torch.max(logit, dim=0)[1]) + 1
            
            msg += "{}\n".format(pred)
        
        with open(args.save_root + video_id + ".txt", "w") as ftxt:
            ftxt.write(msg)

        pbar.update(1)

    pbar.close()