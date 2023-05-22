import csv
import os

from cfg import get_args

def extract_info(hitter_path, threshold=80):
    with open(hitter_path, "r") as ftxt:
        datas = ftxt.read().split('\n')

    preds = {}
    n_frames = 0
    for data in datas:
        if data == '':
            continue
        frame_id, pred_cls, pred_score = data.split(' ')
        frame_id = int(frame_id)
        pred_score = float(pred_score)
        if pred_score < threshold:
            pred_cls = 'None'
        preds[frame_id] = [pred_cls, pred_score]
        if frame_id > n_frames:
            n_frames = frame_id

    for i in range(n_frames + 1):
        if i not in preds:
            preds[i] = ['None', 0.0]

    for i in range(1, n_frames):
        if preds[i+1][0] == 'None' and preds[i-1][0] == 'None':
            preds[i][0] = 'None'
    
    return preds, n_frames



def simple_process(preds: dict, n_frames: int, path: str):

    for i in range(n_frames + 1 - 3):
        if i not in preds:
            if i+1 in preds and preds[i+1][0] != 'None':
                preds[i] = [preds[i+1][0], 0.0]
            elif i+2 in preds and preds[i+2][0] != 'None':
                preds[i] = [preds[i+2][0], 0.0]
            else:
                preds[i] = ['None', 0.0]
            continue

        if i+1 not in preds or i+2 not in preds or i+3 not in preds:
            # print(preds, n_frames)
            print(path)

        if preds[i][0] == 'None':
            if preds[i+1][0] != 'None':
                preds[i][0] = preds[i+1][0]
            elif preds[i+2][0] != 'None':
                preds[i][0] = preds[i+2][0]
            elif preds[i+3][0] != 'None':
                preds[i][0] = preds[i+3][0]
        else:
            if preds[i+1][0] == 'None' and preds[i+2][0] == 'None' and preds[i+3][0] == 'None':
                preds[i][0] = 'None'
    return preds


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
                    if infos[prvs_i][1] + 1.5 * vecs[prvs_i][1] < 0:
                        continue
                if abs(infos[prvs_i][0] - infos[next_i][0]) > 150 or abs(infos[prvs_i][1] - infos[next_i][1]) > 200:
                    continue
                _x = fill_value(prvs_i, next_i, i, infos[prvs_i][0], infos[next_i][0])
                _y = fill_value(prvs_i, next_i, i, infos[prvs_i][1], infos[next_i][1])
                # _s = fill_value(prvs_i, next_i, i, infos[prvs_i][2], infos[next_i][2])
                _s = -1
                if _x is None or _y is None:
                    continue
                infos[i] = [_x, _y, _s]

    return infos



def interpolte(areas: list):
    remove_ids = set([])
    for i in range(len(areas) - 1):
        if areas[i+1][1] - areas[i][1] < 2:
            remove_ids.add(i)
            remove_ids.add(i+1)

    _r = 0
    for i in remove_ids:
        del areas[i-_r]
        _r += 1

    remove_ids = set([])
    for i in range(len(areas) - 1):
        if areas[i][0] == areas[i+1][0]:
            if areas[i][3] < 60 and areas[i+1][3] < 70:
                remove_ids.add(i)
                continue

    _r = 0
    for i in remove_ids:
        del areas[i-_r]
        _r += 1

    # ==== 
    remove_ids = set([])
    if len(areas) > 2:
        if areas[-1][0] == areas[-2][0]:
            if areas[-1][1] - areas[-2][1] < 20:
                if areas[-1][3] - areas[-2][3] < -8 and areas[-2][3] - areas[-3][3] > 0: 
                    # precision should gradual descent
                    print("precision up...")
                    remove_ids.add(len(areas) - 2)

    _r = 0
    for i in remove_ids:
        del areas[i-_r]
        _r += 1

    ## if AA or BB
    insert_info = []
    for i in range(len(areas)-1):
        if areas[i][0] == areas[i+1][0]:
            if areas[i+1][0] == 'A':
                insert_hitter = 'B'
            else:
                insert_hitter = 'A'
            prvs_frame = areas[i][2][1]
            next_frame = areas[i+1][2][1]
            mid = (prvs_frame + next_frame) / 2
            insert_info.append([
                i+1,
                [insert_hitter, mid, [prvs_frame+1, next_frame-1], None]
            ])

    for info in insert_info:
        areas.insert(info[0], info[1])

    # if len(areas) > 2:
    #     if float(areas[-1][-1]) > 99.75:
    #         print("WOW")

    remove_ids = set([])
    for i in range(len(areas) - 1):
        if areas[i+1][1] - areas[i][1] < 2:
            remove_ids.add(i)
            remove_ids.add(i+1)

    _r = 0
    for i in remove_ids:
        del areas[i-_r]
        _r += 1

    return areas

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


def check_first(ball_path, obj_path, first_hit_frame):
    balls, max_frames = read_ball_txt(ball_path)
    objs = load_objs(obj_path, max_frames)
    status = []
    for i in range(0, max_frames):
        if i > int(first_hit_frame) - 1:
            break
        if i not in balls:
            continue
        if balls[i][-1] == -1:
            continue
        if balls[i][0] < objs[i]['net'][0] - 0.5 * objs[i]['net'][2]:
            continue
        if balls[i][0] > objs[i]['net'][0] + 0.5 * objs[i]['net'][2]:
            continue
        min_y = objs[i]['net'][1] - 0.5 * objs[i]['net'][3]
        ball_y = balls[i][1]
        if ball_y < min_y:
            status.append(-1)
        else:
            status.append(1)

    changing_ids = []
    for i in range(len(status) - 1):
        if status[i] * status[i+1] < 0:
            changing_ids.append(i)

    if len(changing_ids) == 1:
        # print('    ', ball_path, changing_ids)
        if changing_ids[0] + 1 > 3 and len(status) - changing_ids[0] > 5:
            for i in range(0, int(first_hit_frame) - 1):
                if i not in balls or i + 1 not in balls:
                    continue
                dis = ((balls[i+1][0] - balls[i][0]) ** 2 + (balls[i+1][1] - balls[i][1]) ** 2) ** 0.5
                if dis > 5:
                    return i

    return None

def build_hitters_list(preds: dict, n_frames: int):
    prvs_hit = 'Z'
    start_frame = None
    areas = []
    for i in range(n_frames + 1):
        if i not in preds:
            continue
        if preds[i][0] != prvs_hit:
            if start_frame is not None:
                if prvs_hit != "None":
                    if i - start_frame > 4:
                        mid = (start_frame + i) / 2
                        avg_score = 0
                        for j in range(start_frame, i):
                            avg_score += preds[j][1]/(i - start_frame)
                        areas.append([prvs_hit, mid, [start_frame, i], avg_score])
            start_frame = i
            prvs_hit = preds[i][0]
        
    return areas

def save_dict(save_path, preds: dict, n_frames: int):
    msg = ''
    for i in range(n_frames + 1):
        if i in preds:
            msg += '{} {}\n'.format(i, preds[i])
        else:
            msg += '{} None\n'.format(i)
    with open(save_path, "w") as ftxt:
        ftxt.write(msg)

def save_list(save_path, areas: list):
    msg = ''
    for i in range(len(areas)):
        hitter = areas[i][0]
        hit_frame = areas[i][1]
        hit_range_start = areas[i][2][0]
        hit_range_end = areas[i][2][1]
        avg_score = areas[i][3]
        msg += '{} {} [{} {}] {}\n'.format(hitter, hit_frame, hit_range_start, hit_range_end, avg_score)
        
    with open(save_path, "w") as ftxt:
        ftxt.write(msg)

if __name__ == "__main__":

    args=get_args()
    threshold = 0
    file_root = "./result_hitters_{}_last_pth/".format(args.prefix) # 
    filtered_file_root = "./result_hitters_seq_filtered_{}/".format(args.prefix)
    save_root = "../results/result_hitters_frame_{}/".format(args.prefix)
    os.makedirs(filtered_file_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)
    
    # ball_root = "../ball_locs_filter_test/" # for test
    # ball_root = ""
    
    # obj_root = '../part2_loc_filtered/' # for test

    # label_root = "../../DATA/part1/train/"
    label_root = None
    
    corrects, total = 0, 0
    accuracy_msg = ""
    files = set(os.listdir(file_root))
    for i in range(6, 7):
        idx = str(i).zfill(5)
        file_name = "{}_{}".format(args.prefix, idx)
        if file_name + ".txt" not in files:
            continue
        print(idx)
        # print("...{}".format(idx))

        file_path = file_root + file_name + ".txt"
        preds, n_frames = extract_info(file_path, threshold)
        filtered_preds = simple_process(preds, n_frames, file_path)
        save_dict(filtered_file_root + file_name + ".txt", filtered_preds, n_frames)

        areas = build_hitters_list(filtered_preds, n_frames)


        areas = interpolte(areas)

        add_first_hit_frame = check_first(ball_path=args.ball_root + file_name + ".txt", 
                                          obj_path=args.obj_root + file_name, 
                                          first_hit_frame=areas[0][1])

        if add_first_hit_frame is not None:
            print(file_name)
            if areas[0][0] == 'A':
                add_hitter = 'B'
            else:
                add_hitter = 'A'
            areas = [[add_hitter, add_first_hit_frame, [None, None], None]] + areas
        
        save_list(save_root + file_name + ".txt", areas)

        ### compare with label
        if label_root is None or len(label_root) == "":
            continue

        label_path = label_root + idx + "/" + idx + "_S2.csv"
        rows = []
        with open(label_path, 'r') as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                rows.append(row)
        gt_count = len(rows) - 1

        if gt_count == len(areas):
            corrects += 1
            isCorrect = True
        else:
            isCorrect = False
            print("    ", file_path)
            print("ground truth:", gt_count, "pred:", len(areas))
            print()

        accuracy_msg += "video:{}, pred:{}, lb:{}, isCorrect:{}\n".format(file_name, len(areas), gt_count, isCorrect)
        total += 1


    if label_root is not None:
        print(corrects / total)

        with open("acc.txt", "w") as ftxt:
            ftxt.write(accuracy_msg)
