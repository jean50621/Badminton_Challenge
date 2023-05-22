import numpy as np
import pandas as pd
from numpy import random
import time
from pathlib import Path
import cv2
import os
import tqdm
import json

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from cfg import get_args

def RTMPose(img_org, video_name, model, players_arr, save_path, player_loc):

    Location = {}
    for idx, frame in enumerate(players_arr):
        R_foot = []
        L_foot = []

        
        for i in range(len(players_arr[frame])):

            result = inference_topdown(model, players_arr[frame][i])
            keps = result[0].pred_instances['keypoints'][0]

            x_min = player_loc[str(frame)][i][0]
            y_min = player_loc[str(frame)][i][1]

            Rfootx = keps[21][0]+x_min
            Rfooty = keps[21][1]+y_min
            R_foot.extend([[Rfootx, Rfooty]])

            Lfootx = keps[18][0]+x_min
            Lfooty = keps[18][1]+y_min
            L_foot.extend([[Lfootx, Lfooty]])
            

            arr = img_org[frame]

            h, w = arr.shape[:2] # 384, 288
            for kep in keps[17:23]:
                x, y = kep[0], kep[1]
                plot_x_y(arr, x+x_min, y+y_min)

            for kep in keps[0:17]:
                x, y = kep[0], kep[1]
                plot_x_y(arr, (x+x_min), (y+y_min), color=(255, 0, 0))


        tmp = compare(R_foot, L_foot, frame, video_name)
        Location[str(frame-1)] = tmp

    save_csv(Location, save_path, video_name)

    


def compare(R_foot, L_foot, frame, video_name):
    ARLocation = []
    BRLocation = []
    ALLocation = []
    BLLocation = []

    # A or B
    if len(R_foot) == 1 or len(L_foot) == 1:
 
        BRLocation.append(R_foot[0][0])
        BRLocation.append(R_foot[0][1])
        BLLocation.append(L_foot[0][0])
        BLLocation.append(L_foot[0][1])

    elif (R_foot[0][1]) < (R_foot[1][1]) or (L_foot[0][1]) < (L_foot[1][1]):

        ARLocation.append(R_foot[0][0])
        ARLocation.append(R_foot[0][1])
        BRLocation.append(R_foot[1][0])
        BRLocation.append(R_foot[1][1])
        ALLocation.append(L_foot[0][0])
        ALLocation.append(L_foot[0][1])
        BLLocation.append(L_foot[1][0])
        BLLocation.append(L_foot[1][1])

    else:

        ARLocation.append(R_foot[1][0])
        ARLocation.append(R_foot[1][1])
        BRLocation.append(R_foot[0][0])
        BRLocation.append(R_foot[0][1])
        ALLocation.append(L_foot[1][0])
        ALLocation.append(L_foot[1][1])
        BLLocation.append(L_foot[0][0])
        BLLocation.append(L_foot[0][1])


    tmp = {
        "A": {
                "right":
                    ARLocation,
                "left":
                    ALLocation,    
            },
            "B": {
                "right":
                    BRLocation,    
                "left":
                    BLLocation,
            }

    }

    return tmp

def save_csv(Location: dict, save_path: str, video_name: str):
    p = str(save_path) + "/" + "{}.json".format(video_name)
    with open(p, "w") as f:
        f.write(json.dumps(Location,indent=2))


def plot_x_y(arr, x, y, color=(0, 0, 255)):
    px, py = int(x), int(y)
    cv2.circle(arr,(px, py), 3, color, -1)
    #cv2.imshow("pose", arr)

def read_loc_txt(loc_path, img_h, img_w):
    loc_files = os.listdir(loc_path)
    player_loc = {}
    for txt_file in loc_files :
        frame_id = txt_file.split('.')[0]
        player_loc[frame_id] = []
        with open(loc_path + txt_file, 'r') as ftxt:
            datas = ftxt.read().split('\n')
        for data in datas :
            if data == '':
                continue
            infos = data.split(' ')
            if int(infos[0]) == 0:
                x = float(infos[1]) * img_w
                y = float(infos[2]) * img_h
                w = float(infos[3]) * img_w
                h = float(infos[4]) * img_h
                x_min = x - w*0.5
                y_min = y - h*0.5
                x_max = x + w*0.5
                y_max = y + h*0.5

                player_loc[frame_id].append([x_min, y_min, x_max, y_max])

    return player_loc

def crop(video_name, img_root, player_loc):

    players = {}
    img_org = {}
    img_path = img_root + 'val_{}'.format(video_name)
    img_files = os.listdir(img_path)

    for img_file in img_files:

        img_name = img_file.split('.')[0]
        img = cv2.imread(img_path + '/' + img_file)
        img_org[int(img_name)+1] = img
        for idx, frame_id in enumerate(player_loc):
            if int(frame_id) == int(img_name)+1 :
                players[int(frame_id)] = []
                for k in range(len(player_loc[frame_id])):
                    x0 = player_loc[frame_id][k][0]
                    y0 = player_loc[frame_id][k][1]
                    x1 = player_loc[frame_id][k][2]
                    y1 = player_loc[frame_id][k][3]
                    arr = img[int(y0):int(y1), int(x0):int(x1)]
                    players[int(frame_id)].append(arr)

    return players, img_org




if __name__ == '__main__':

    args = get_args()

    save_path = Path(args.save_root)
    save_path.mkdir(parents=True, exist_ok=True)

    register_all_modules()
    config_file = 'rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
    checkpoint_file = 'rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-384x288-eaeb96c8_20230125.pth'
    model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'


    loc_files = os.listdir(args.loc_root)
    pbar = tqdm.tqdm(total=len(loc_files), ascii=True)
    for loc_file in loc_files:
        loc_path = args.loc_root + loc_file + '/'
        video_name = loc_file.split('_')[1]
        print("video_name:", video_name)
        player_loc = read_loc_txt(loc_path, 720, 1280)
        players_arr, img_org = crop(video_name, args.img_root, player_loc)
        RTMPose(img_org, video_name, model, players_arr, save_path, player_loc)

        pbar.update(1)
    pbar.close()



    