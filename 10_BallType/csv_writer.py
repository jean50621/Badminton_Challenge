import csv
import os
import random
from cfg import get_args

def read_ball_txt(path: str) -> [dict, int]:
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

    vecs = calculate_vecs(infos, max_frames)
    infos = interplote(infos, vecs, max_frames)
    vecs = calculate_vecs(infos, max_frames)
    noise_frames = filter_out_first_noise(vecs, max_frames)
    if len(noise_frames) > 0 and noise_frames[0] < 60: # first two secs
        # remove all results before noise_frames[0]
        for i in range(noise_frames[0], -1, -1):
            if i in infos:
                del infos[i]
    return infos, max_frames

def find_prvs(infos: dict, i: int):
    for j in range(i-1, -1, -1):
        if j in infos:
            return j
    return None

def find_next(infos: dict, i: int, max_frames: int):
    for j in range(i+1, max_frames+1):
        if j in infos:
            return j
    return None


def fill_value(prvs_i: int, next_i: int, target_i: int, prvs_val: float, next_val: float) -> float:
    if (next_val - prvs_val) > 200:
        return None
    gap = next_i - prvs_i
    prop = (target_i - prvs_i) / gap
    diff = next_val - prvs_val
    target_val = prvs_val + diff * prop
    return target_val

def calculate_vecs(infos: dict, max_frames: int):
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

def filter_out_first_noise(vecs: dict, max_frames: int):
    noise_frames = []
    for i in range(max_frames):
        if i in vecs:
            if abs(vecs[i][0]) > 80 or abs(vecs[i][1]) > 100:
                noise_frames.append(i)
    return noise_frames


def interplote(infos: dict, vecs: dict, max_frames: int) -> dict:
    for i in range(max_frames + 1):
        if i not in infos:
            prvs_i = find_prvs(infos, i)
            next_i = find_next(infos, i, max_frames)
            if prvs_i is not None and next_i is not None:
                if prvs_i in vecs:
                    if infos[prvs_i][1] + vecs[prvs_i][1] < 0:
                        continue
                if abs(infos[prvs_i][0] - infos[next_i][0]) > 150 or abs(infos[prvs_i][1] - infos[next_i][1]) > 200:
                    continue
                _x = fill_value(prvs_i, next_i, i, infos[prvs_i][0], infos[next_i][0])
                _y = fill_value(prvs_i, next_i, i, infos[prvs_i][1], infos[next_i][1])
                _s = fill_value(prvs_i, next_i, i, infos[prvs_i][2], infos[next_i][2])
                if _x is None or _y is None:
                    continue
                infos[i] = [_x, _y, _s]

    return infos



def load_objs(root: str, n_frames: int):
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


if __name__ == "__main__":

    args = get_args()


    ball_root = args.ball_root
    obj_root = args.obj_root

    result_root = args.save_root
    files = os.listdir(result_root)

    datas_to_write = {}
    datas_first_random = {}
    for file in files:
        file_name = file.split('.')[0] + ".mp4"
        datas_to_write[file_name] = []
        with open(result_root + file, 'r') as ftxt:
            datas = ftxt.read().split('\n')
        for data in datas:
            if data == '':
                continue
            if len(datas_to_write[file_name]) == 0:
                if not (int(data) == 1 or int(data) == 2):
                    data = random.randint(1, 2)
                    print("err [1]", file_name, int(data))
                    datas_first_random[file_name] = 1
            else:
                if int(data) == 1:
                    data = 8
                    print("err [2]", file_name)
                elif int(data) == 2:
                    data = 3
                    print("err [3]", file_name)

            datas_to_write[file_name].append(int(data))

    rows = []
    with open("../aicup_final_{}.csv".format(args.prefix), 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    video_data_count = {}
    prvs_video_name = None
    for i in range(len(rows)):
        if i == 0:
            continue
        video_name = rows[i][0]
        # print(video_name, video_name in datas_to_write)
        assert video_name in datas_to_write
        if video_name not in video_data_count:
            video_data_count[video_name] = 0
            if video_name in datas_first_random:
                if prefix == 'val':
                    t_thre = 10
                else:
                    t_thre = 30
                if i+1 < len(rows) and rows[i+1][0] == video_name:
                    # print(video_name, rows[i][2], rows[i+1][2], int(rows[i+1][2]) - int(rows[i][2]))
                    if int(rows[i+1][2]) - int(rows[i][2]) > 35:
                        datas_to_write[video_name][video_data_count[video_name]] = 2
                        print("[1]->", video_name, 2)
                    
                    elif int(rows[i+1][2]) - int(rows[i][2]) >= t_thre: # 10~35 # check player locations
                        obj_path = obj_root + "{}_".format(prefix) + video_name.split('.')[0] + "/"
                        n_frames = len(os.listdir(obj_path)) + 1
                        objs = load_objs(obj_path, n_frames)
                        A_loc = objs[int(rows[i+1][2])]['player']['A']
                        B_loc = objs[int(rows[i+1][2])]['player']['B']
                        dis = ((A_loc[0] - B_loc[0]) ** 2 + (A_loc[1] - B_loc[1]) ** 2) ** 0.5
                        if dis > 220:
                            datas_to_write[video_name][video_data_count[video_name]] = 2
                            print("[2]->", video_name, 2, dis)
                        else:
                            datas_to_write[video_name][video_data_count[video_name]] = 1
                            print("[3]->", video_name, 1, dis)
                    
                    elif int(rows[i+1][2]) - int(rows[i][2]) < t_thre:
                        datas_to_write[video_name][video_data_count[video_name]] = 1
                        print("[4]->", video_name, 1, int(rows[i+1][2]) - int(rows[i][2]))
                else:
                    ball_path = ball_root + "{}_".format(prefix) + video_name.split('.')[0] + ".txt"
                    balls, n_frames = read_ball_txt(ball_path)
                    ys = []
                    for ball_i in range(int(rows[i][2]), n_frames):
                        if ball_i not in balls:
                            continue
                        ys.append(balls[ball_i][1] )
                    dis_y = 0
                    for yi in range(len(ys) - 1):
                        dis_y += abs(ys[yi + 1] - ys[yi])
                    if dis_y > 300:
                        datas_to_write[video_name][video_data_count[video_name]] = 2
                        print("[5]->", video_name, 2)
                    else:
                        datas_to_write[video_name][video_data_count[video_name]] = 1
                        print("[6]->", video_name, 1)

        _res = datas_to_write[video_name][video_data_count[video_name]]
        video_data_count[video_name] += 1

        rows[i][13] = _res

        prvs_video_name = rows[i][0]

    with open("../aicup_final_{}.csv".format(args.prefix), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
