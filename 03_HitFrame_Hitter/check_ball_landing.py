import os


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


if __name__ == "__main__":
    prefix = "val"
    if prefix == "test":
        start_idx = 170
        end_idx = 400
    else:
        start_idx = 6
        end_idx = 7
    remove_number = 15
    count = 0
    for idx in range(start_idx, end_idx):
        video_name = "{}_{}".format(prefix, str(idx).zfill(5))
        hitframe_path = "./result_hitters_frame_{}/".format(prefix) + video_name + ".txt"
        
        if prefix == "test":
            ball_root = '../ball_locs_filter_test/{}.txt'.format(video_name)
        else:
            ball_root = '../ball_loc_results_filtered/{}.txt'.format(video_name)
        
        if prefix != "test":
            yolo_net_root = '../player_location/{}/'.format(video_name) # for train and validation set.
        else:
            yolo_net_root = '../loc_filtered/{}/'.format(video_name)

        with open(hitframe_path, "r") as ftxt:
            datas = ftxt.read().split('\n')

        data = datas[-2].split(' ')
        # print(data)
        # print(data[1])

        balls, max_frames = read_ball_txt(ball_root)
        # print(len(balls))

        
        start_frames = int(float(data[1]))
        end_frames = max_frames
        tmps = []
        for i in range(start_frames, end_frames):
            if i in balls:
                tmps.append([balls[i][0], balls[i][1]])

        total_dis = 0
        count = 0
        for i in range(len(tmps)-1):
            dis = ((tmps[i+1][0] - tmps[i][0]) ** 2 + (tmps[i+1][1] - tmps[i][1]) ** 2) ** 0.5
            total_dis += dis
            count += 1
        
        print(idx, total_dis)
        
        ### remove_number

        if prefix == "test":
            dis_thre = 75
        else:
            dis_thre = 80
        
        if total_dis < dis_thre:
            count += 1
            with open(hitframe_path, "r") as ftxt:
                datas = ftxt.read().split('\n')
            del datas[-2]
            msg = ""
            for data in datas:
                if data == '':
                    continue
                msg += data + "\n"
            with open(hitframe_path, "w") as ftxt:
                ftxt.write(msg)

    print(count)
