import os
import csv
from tqdm import tqdm

import json
from cfg import get_args


def get_video_skeleton_from_csv(root, video_idx, pbar):
    files = os.listdir(root)
    skeletons = {}
    for file in files:
        pbar.set_description("process... {}-{}".format(video_idx, file))
        if file.split("_")[0] == video_idx:
            frame_n = int(file.split("_")[1].split('.')[0]) - 1
            skeletons[frame_n] = read_csv_file(root + file)
    return skeletons


def get_video_skeleton_from_json(root, video_idx):
    with open(root + video_idx + ".json", "r") as fjson:
        skeletons = json.load(fjson)
    return skeletons



def read_csv_file(csv_path):
    rows = []
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    A_rfoot = [None, None]
    A_lfoot = [None, None]
    B_rfoot = [None, None]
    B_lfoot = [None, None]
    for ri, row in enumerate(rows):
        if ri == 0:
            if 'Empty' in row[0]:
                break
        if ri == 1:
            infos = [x for x in row[0].split(' ') if x not in ['', ' ']]
            infos = [float(x) for x in infos[1:]]
            A_rfoot = [infos[0], infos[1]] # A right foot X, Y
            B_rfoot = [infos[2], infos[3]]
            A_lfoot[0] = infos[4]
        if ri == 4:
            infos = [x for x in row[0].split(' ') if x not in ['', ' ']]
            infos = [float(x) for x in infos[1:]]
            A_lfoot[1] = infos[0]
            B_lfoot = [infos[1], infos[2]]

    foots = {
        'A':{'right':A_rfoot, 'left':A_lfoot},
        'B':{'right':B_rfoot, 'left':B_lfoot},
    }

    return foots

def read_txt(root, video_idx):
    with open(root + video_idx + ".txt", "r") as ftxt:
        datas = ftxt.read().split('\n')

    preds = {}
    for data in datas:
        if data == '':
            continue
        infos = data.split(' ')

        hit_frame = int(infos[0])
        preds[hit_frame] = {
            'hitter':infos[1],
            'hitter_x':float(infos[2]),
            'hitter_y':float(infos[3]),
            'pred_hitter_x':float(infos[4]),
            'pred_hitter_y':float(infos[5]),
            'def_x':float(infos[6]),
            'def_y':float(infos[7]),
            'pred_def_x':float(infos[8]),
            'pred_def_y':float(infos[9]),
        }

    return preds


def l2dis(loc1, loc2):

    dis = ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    return dis


def find_handedness(pred_x, pred_y, player_skeleton):
    dis_right = l2dis([pred_x, pred_y], player_skeleton['right'])
    dis_left = l2dis([pred_x, pred_y], player_skeleton['left'])
    if dis_right < dis_left:
        return 'RIGHT'
    else:
        return 'LEFT'


def calibrate(hitter, pred_x, pred_y, player_skeleton, handednes=None):
    # 1. check minimum distance
    if handednes is None:
        raise ValueError('handednes should not be None')
        dis_right = l2dis([pred_x, pred_y], player_skeleton['right'])
        dis_left = l2dis([pred_x, pred_y], player_skeleton['left'])
        if dis_right < dis_left:
            handednes = 'RIGHT'
        else:
            handednes = 'LEFT'

    
    if handednes == 'RIGHT':
        if player_skeleton['right'][0] is not None:
            x = player_skeleton['right'][0]
            if pred_y > player_skeleton['right'][1]:
                y = (player_skeleton['right'][1] + pred_y * 3) / 4
            else:
                y = (player_skeleton['right'][1] * 3 + pred_y) / 4
            px = player_skeleton['right'][0]
            py = player_skeleton['right'][1]
        else:
            x = pred_x
            y = pred_y
            px = None
            py = None
    else:
        if player_skeleton['left'][0] is not None:
            x = player_skeleton['left'][0]
            if pred_y > player_skeleton['left'][1]:
                y = (player_skeleton['left'][1] + pred_y * 3) / 4
            else:
                y = (player_skeleton['left'][1] * 3 + pred_y) / 4
            px = player_skeleton['left'][0]
            py = player_skeleton['left'][1]
        else:
            x = pred_x
            y = pred_y
            px = None
            py = None

    return x, y, px, py


def find_notNone(skeletons, hit_frame, max_search_number = 4):
    next_frame = None
    for i in range(hit_frame, hit_frame + max_search_number + 1):
        if str(i) in skeletons and skeletons[str(i) ]['A']['right'][0] is not None:
            next_frame = i
            break

    prvs_frame = None
    for i in range(hit_frame - max_search_number, hit_frame):
        if str(i)  in skeletons and skeletons[str(i)]['A']['right'][0] is not None:
            prvs_frame = i
            break

    if next_frame is not None and prvs_frame is not None:
        if abs(next_frame - hit_frame) < abs(prvs_frame - hit_frame):
            return next_frame
        else:
            return prvs_frame
    elif next_frame is not None:
        return next_frame
    elif prvs_frame is not None:
        return prvs_frame

    return None


def matching(preds, skeletons, skeleton_root, idx, pbar):

    results = {}

    max_frames = None

    handednes = {}
    
    for hit_frame in preds:

        pbar.set_description("...{}/{} {}".format(skeleton_root, idx, hit_frame))

        pred_infos = preds[hit_frame]
        # assert str(hit_frame) in skeletons, "{}/{} {}".format(skeleton_root, idx, hit_frame)


        if str(hit_frame) not in skeletons or skeletons[str(hit_frame)]['A']['right'][0] is None:
            _hit_frame = find_notNone(skeletons, hit_frame, 50) # val only need 5
        else:
            _hit_frame = hit_frame

        pose_infos = skeletons[str(_hit_frame)]

        results[hit_frame] = {}
        
        # 1. process hitter
        hitter = pred_infos['hitter']
        pred_x = pred_infos['pred_hitter_x']
        pred_y = pred_infos['pred_hitter_y']
        player_skeleton = pose_infos[hitter]

        _hand = find_handedness(pred_x, pred_y, player_skeleton)

        if hitter not in handednes:
            handednes[hitter] = []
        handednes[hitter].append(_hand)

        # 2. process defender
        if pred_infos['hitter'] == 'A':
            hitter = 'B'
        else:
            hitter = 'A'
        pred_x = pred_infos['pred_def_x']
        pred_y = pred_infos['pred_def_y']
        player_skeleton = pose_infos[hitter]

        _hand = find_handedness(pred_x, pred_y, player_skeleton)

        if hitter not in handednes:
            handednes[hitter] = []
        handednes[hitter].append(_hand)


    voted_handednes = {}
    for name in handednes:
        votes = [0, 0]
        for _h in handednes[name]:
            if _h == 'RIGHT':
                votes[0] += 1
            else:
                votes[1] += 1

        if votes[0] > votes[1]:
            voted_handednes[name] = 'RIGHT'
        elif votes[1] > votes[0]:
            voted_handednes[name] = 'LEFT'
        else:
            voted_handednes[name] = 'RIGHT'

    # print(idx, voted_handednes)

    handednes = voted_handednes

    for hit_frame in preds:    

        pbar.set_description("[!!]...{}/{} {}".format(skeleton_root, idx, hit_frame))

        pred_infos = preds[hit_frame]
        # assert str(hit_frame) in skeletons, "{}/{} {}".format(skeleton_root, idx, hit_frame)

        if str(hit_frame) not in skeletons or skeletons[str(hit_frame)]['A']['right'][0] is None:
            _hit_frame = find_notNone(skeletons, hit_frame, 50)
        else:
            _hit_frame = hit_frame

        pose_infos = skeletons[str(_hit_frame)]

        results[hit_frame] = {}
        
        # 1. process hitter
        hitter = pred_infos['hitter']
        pred_x = pred_infos['pred_hitter_x']
        pred_y = pred_infos['pred_hitter_y']
        player_skeleton = pose_infos[hitter]

        _x, _y, _px, _py = calibrate(hitter, pred_x, pred_y, player_skeleton, handednes[hitter])
        results[hit_frame]['hitter'] = [_px, _py, _x, _y]
        
        # 2. process defender
        if pred_infos['hitter'] == 'A':
            hitter = 'B'
        else:
            hitter = 'A'
        pred_x = pred_infos['pred_def_x']
        pred_y = pred_infos['pred_def_y']
        player_skeleton = pose_infos[hitter]
        _x, _y, _px, _py = calibrate(hitter, pred_x, pred_y, player_skeleton, handednes[hitter])
        results[hit_frame]['defender'] = [_px, _py, _x, _y]

        if max_frames is None or hit_frame > max_frames:
            max_frames = hit_frame

    return results, max_frames + 1


def write_results(save_path, results, max_frames):
    msg = ""
    for i in range(max_frames):
        if i not in results:
            continue
        msg += "{} {} {} {} {} ".format(i, results[i]['hitter'][0], results[i]['hitter'][1],
            results[i]['hitter'][2], results[i]['hitter'][3]
            )
        msg += "{} {} {} {}\n".format(results[i]['defender'][0], results[i]['defender'][1],
            results[i]['defender'][2], results[i]['defender'][3]
            )

    with open(save_path, "w") as ftxt:
        ftxt.write(msg)


if __name__ == "__main__":

    args= get_args()
    skeleton_root = args.skeleton_root # "./positions/"
    # save_root = "./skeleton_jsons/"
    save_root = args.filtered_root
    os.makedirs(save_root, exist_ok=True)
    preds_root = args.preds_root

    pbar = tqdm(total=1, ascii=True)
    for i in range(6, 7):
        video_idx = str(i).zfill(5) # "00001"
        pbar.set_description("process... {}".format(video_idx))

        skeletons = get_video_skeleton_from_json(skeleton_root, video_idx)
        preds = read_txt(preds_root, video_idx)

        results, max_frames = matching(preds, skeletons, skeleton_root, video_idx, pbar)
        write_results(save_root + video_idx + ".txt", results, max_frames)

        pbar.update(1)

