import os
from tqdm import tqdm
import cv2
from copy import deepcopy
import numpy as np
from cfg import get_args

#from yolo import get_model, predict

def IoU(box1, box2):

    boxA = deepcopy(box1)
    boxB = deepcopy(box2)
    
    boxA[0] *= 1280
    boxA[1] *= 720
    boxA[2] *= 1280
    boxA[3] *= 720

    _boxA = [0, 0, 0, 0]
    _boxA[0] = boxA[0] - 0.5 * boxA[2]
    _boxA[1] = boxA[1] - 0.5 * boxA[3]
    _boxA[2] = boxA[0] + 0.5 * boxA[2]
    _boxA[3] = boxA[1] + 0.5 * boxA[3]
    boxA = _boxA


    boxB[0] *= 1280
    boxB[1] *= 720
    boxB[2] *= 1280
    boxB[3] *= 720

    _boxB = [0, 0, 0, 0]
    _boxB[0] = boxB[0] - 0.5 * boxB[2]
    _boxB[1] = boxB[1] - 0.5 * boxB[3]
    _boxB[2] = boxB[0] + 0.5 * boxB[2]
    _boxB[3] = boxB[1] + 0.5 * boxB[3]
    boxB = _boxB

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def read_objs_txt(txt_path):
    with open(txt_path, "r") as ftxt:
        datas = ftxt.read().split('\n')

    cls_table = {0:'player', 1:'net', 2:'court'}


    objs = {'player':[], 'net':[], 'court':[]}
    for data in datas:
        if data == '':
            continue
        info = data.split(' ')
        x, y, w, h = [float(x) for x in info[1:]]
        cls_type = int(info[0])
        cls_name = cls_table[cls_type]
        objs[cls_name].append([x, y, w, h])

    if len(objs['net']) < 1:
        return None

    ### get one net
    max_area = 0
    net = None
    if len(objs['net']) > 1:
        for n in objs['net']:
            x, y, w, h = n
            if x - 0.5 * w < 0.05:
                continue
            a = n[2] * n[3]
            if a > max_area:
                max_area = a
                net = n
        objs['net'] = net
    else:
        objs['net'] = objs['net'][0]
    

    if len(objs['player']) != 2:
        xl, xu = objs['net'][0] - 0.55 * objs['net'][2], objs['net'][0] + 0.55 * objs['net'][2]
        players = []
        for p in objs['player']:
            if xl <= p[0] <= xu:
                players.append(p)
        # print(len(objs['player']), "->", len(players))
        objs['player'] = players 

    if len(objs['player']) > 2:
        del_idx = None
        for i in range(len(objs['player'])):
            objs1 = objs['player'][i]
            x1, y1, w1, h1 = objs1
            if (w1 > objs['net'][2] / 2.5):
                del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]

    if len(objs['player']) > 2:
        del_idx = None
        for i in range(len(objs['player'])):
            objs1 = objs['player'][i]
            x1, y1, w1, h1 = objs1
            for j in range(len(objs['player'])):
                if i == j:
                    continue
                objs2 = objs['player'][j]
                x2, y2, w2, h2 = objs2 #  - 0.0039, - 0.0069
                if x1 + 0.5 * w1 < x2 + 0.5 * w2 and x1 - 0.5 * w1 > x2 - 0.5 * w2:
                    if y1 + 0.5 * h1 < y2 + 0.5 * h2 and y1 - 0.5 * h1 > y2 - 0.5 * h2:
                        del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]

    if len(objs['player']) > 2:
        del_idx = None
        for i in range(len(objs['player'])):
            objs1 = objs['player'][i]
            x1, y1, w1, h1 = objs1
            if w1 * h1 < 0.003:
                del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]

    if len(objs['player']) > 2:
        IoUs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(len(objs['player'])):
            _msg = "    "
            for j in range(len(objs['player'])):
                v = IoU(objs['player'][i], objs['player'][j])
                _msg += "  {}".format(v)
                IoUs[i][j] = v

        if 0.4 > IoUs[0][1] > 0.1 and 0.4 > IoUs[0][2] > 0.1:
            max_y, min_y = 0, 720
            max_i, min_i = None, None
            for i in range(len(objs['player'])):
                if objs['player'][i][1] < min_y:
                    min_y = objs['player'][i][1]
                    min_i = i
                if objs['player'][i][1] > max_y:
                    max_y = objs['player'][i][1]
                    max_i = i

            new_players = []
            for i in range(len(objs['player'])):
                if i == max_i or i == min_i:
                    new_players.append(objs['player'][i])
            objs['player'] = new_players

    if len(objs['player']) > 2:
        del_idx = None
        for i in range(len(objs['player'])):
            for j in range(len(objs['player'])):
                if i == j:
                    continue
                if abs(objs['player'][i][2] - objs['player'][j][2]) < 0.0078:
                    if abs(objs['player'][i][0] - objs['player'][j][0]) < 0.0078:
                        if objs['player'][i][2] * objs['player'][i][3] > objs['player'][j][2] * objs['player'][j][3]:
                            del_idx = j
                        else:
                            del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]

    if len(objs['player']) > 2:
        del_idx = None
        for i in range(len(objs['player'])):
            objs1 = objs['player'][i]
            x1, y1, w1, h1 = objs1
            for j in range(len(objs['player'])):
                if i == j:
                    continue
                objs2 = objs['player'][j]
                x2, y2, w2, h2 = objs2 #  - 0.0039, - 0.0069
                if x1 + 0.5 * w1 <= x2 + 0.5 * w2 and x1 - 0.5 * w1 >= x2 - 0.5 * w2:
                    if y1 + 0.5 * h1 - 0.015 <= y2 + 0.5 * h2 and y1 - 0.5 * h1 >= y2 - 0.5 * h2:
                        del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]


    if len(objs['player']) > 2:

        IoUs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(len(objs['player'])):
            _msg = "    "
            for j in range(len(objs['player'])):
                v = IoU(objs['player'][i], objs['player'][j])
                _msg += "  {}".format(v)
                IoUs[i][j] = v

        ids = []
        for i in range(3):
            if len(ids) == 2:
                break
            for j in range(3):
                if i == j:
                    continue
                if IoUs[i][j] > 0.4:
                    ids.append(i)
                    ids.append(j)
                    break

        if len(ids) == 2:
            del_idx = None
            if '00172/275' in txt_path:
                print(":\n\n\n", ids, objs['player'], "\n\n\n")
            if abs(objs['player'][ids[0]][0] - objs['player'][ids[1]][0]) < 0.02:
                if abs(objs['player'][ids[0]][3] - objs['player'][ids[1]][3]) < 0.02:
                    if objs['player'][ids[0]][1] * objs['player'][ids[0]][2] > objs['player'][ids[1]][1] * objs['player'][ids[1]][2]:
                        del_idx = ids[0]
                    else:
                        del_idx = ids[1]

            if del_idx is not None:
                del objs['player'][del_idx]


    if len(objs['player']) > 2:
        del_idx = None
        nx, ny, nw, nh = objs['net']
        max_dis = None
        for i in range(len(objs['player'])):
            objs1 = objs['player'][i]
            x1, y1, w1, h1 = objs1
            dis = ((x1 - nx) ** 2 + (x2 - ny) ** 2) ** 0.5
            if max_dis is None or dis > max_dis:
                max_dis = dis
                del_idx = i
        if del_idx is not None:
            del objs['player'][del_idx]


    if len(objs['court']) != 1:
        min_dis = None
        court = None
        for c in objs['court']:
            dis = ((c[0] - objs['net'][0]) ** 2 + (c[1] - objs['net'][1]) ** 2) ** 0.5
            if min_dis is None or dis < min_dis:
                min_dis = dis
                court = c
        objs['court'] = c
    else:
        objs['court'] = objs['court'][0]

    return objs

def plotimage(objs, img_path, save_name, model, IoUs=None):
    arr = cv2.imread(img_path)

    for p in objs['player']:
        x, y, w, h = p
        x0, x1 = int((x - 0.5 * w) * 1280), int((x + 0.5 * w) * 1280)
        y0, y1 = int((y - 0.5 * h) * 720), int((y + 0.5 * h) * 720)
        print("    ", w * h)

        # img = deepcopy(arr[y0:y1, x0:x1])
        # predict(model, img)

        cv2.rectangle(arr, (x0, y0), (x1, y1), (0, 0, 255), 1)

    if len(objs['court']) != 4:
        print("    ", objs['court'])
    else:
        x, y, w, h = objs['court']
        x0, x1 = int((x - 0.5 * w) * 1280), int((x + 0.5 * w) * 1280)
        y0, y1 = int((y - 0.5 * h) * 720), int((y + 0.5 * h) * 720)
        cv2.rectangle(arr, (x0, y0), (x1, y1), (255, 0, 255), 1)

    if len(objs['net']) != 4:
        print("    ", objs['net'])
    else:
        x, y, w, h = objs['net']
        x0, x1 = int((x - 0.5 * w) * 1280), int((x + 0.5 * w) * 1280)
        y0, y1 = int((y - 0.5 * h) * 720), int((y + 0.5 * h) * 720)
        cv2.rectangle(arr, (x0, y0), (x1, y1), (0, 255, 255), 1)

    cv2.imwrite("./errors/" + save_name, arr)


def filter1(objs: dict, file_path, save_name):

    if len(objs['net']) != 4:
        print("ERR 1-0", file_path)
        #plotimage(objs, img_path, save_name, model)

    elif len(objs['player']) > 2:
        print("ERR 1-1", file_path)
        IoUs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(len(objs['player'])):
            _msg = "    "
            for j in range(len(objs['player'])):
                v = IoU(objs['player'][i], objs['player'][j])
                _msg += "  {}".format(v)
                IoUs[i][j] = v
            print(_msg)
        #plotimage(objs, img_path, save_name, model, IoUs)

    # elif len(objs['player']) < 2:
    #     print("ERR 1-2", file_path) # , objs
    #     plotimage(objs, img_path, save_name, model)
    return True


def write_new_txt(path:str,new_info:list): 
    with open(path,'w') as f :
        for x in range(len(new_info)):
            f.write('{} {} {} {} {} \n'.format(new_info[x][0],new_info[x][1],new_info[x][2],new_info[x][3],new_info[x][4]))
        print("Finish write : ",path)

def process_player(path:str,folder_path:str,info:list,count:int):
    file = path.split("/")[-1]
    frame = file.split(".")[0]
    new_info=[]

    if count == 1 :
        
        for i in range(len(info)):
            if info[i][0]!="0":
                new_info.append([info[i][0],info[i][1],info[i][2],info[i][3],info[i][4]])
        
        for j in range(int(float(frame)),1,-1):
            count =0
            find =[]
            process = []
            with open(folder_path+"/{}.txt".format(j)) as f:
                datas = f.read().split('\n')
            datas = [data for data in datas if data != '']

            for data in datas :
                tmp = [None] * 5
                tmp[0] = str(data.split(' ')[0])
                tmp[1] = float(data.split(' ')[1])
                tmp[2] = float(data.split(' ')[2])
                tmp[3] = float(data.split(' ')[3])
                tmp[4] = float(data.split(' ')[4])

                if tmp[0] =='0':
                    find.append([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]])

            if len(find) ==2:
                for k in range(0,2):
                    new_info.append([find[k][0],find[k][1],find[k][2],find[k][3],find[k][4]])

                break
    #print(new_info)
    write_new_txt(path,new_info)

if __name__ == "__main__":
    
    args = get_args()
    #print(args)
    #frame_root = "D:/D-Projects-23/AICUP23/DATA/entir_frames/"
    os.makedirs("../results",exist_ok =True)
    if not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)  
    
    #model = get_model()

    for i in range(6, 7):
        idx = str(i).zfill(5)
        video_name = "val_{}".format(idx)# val or test

        entir_objs = {}
        files = os.listdir(args.root + video_name)
        for i in range(1, len(files) + 1):
            file_path = args.root + video_name + "/{}.txt".format(i)
            objs = read_objs_txt(file_path)
            if objs is not None:
                entir_objs[i] = objs

        filtered_objs = {}
        for i in range(1, len(files) + 1):
            file_path = args.root + video_name + "/{}.txt".format(i)
            #img_path = frame_root + video_name + "/{}.jpg".format(i - 1)
            save_name = video_name + "_{}.jpg".format(i)
            if i not in entir_objs:
                continue
            res = filter1(entir_objs[i], file_path, save_name)
            if res is not None:
                filtered_objs[i] = res
        
        assert len(filtered_objs) == len(entir_objs)

        for i in entir_objs:
            msg = ""
            for name in entir_objs[i]:
                if name == 'player':
                    cls_idx = 0
                    for j in range(len(entir_objs[i][name])):
                        msg += "{} {} {} {} {}\n".format(cls_idx, 
                            entir_objs[i][name][j][0], 
                            entir_objs[i][name][j][1], 
                            entir_objs[i][name][j][2], 
                            entir_objs[i][name][j][3])
                else:
                    if name == 'net':
                        cls_idx = 1
                    if name == 'court':
                        cls_idx = 2
                    msg += "{} {} {} {} {}\n".format(cls_idx, 
                        entir_objs[i][name][0], entir_objs[i][name][1], entir_objs[i][name][2], entir_objs[i][name][3])

            if not os.path.isdir(args.save_root + video_name + "/"):
                os.mkdir(args.save_root + video_name + "/")
            with open(args.save_root + video_name + "/{}.txt".format(i), "w") as ftxt:
                ftxt.write(msg)
    
    #filter2 - player==1
    for folder in os.listdir(args.save_root):

        folder_path = args.save_root+folder
        if not os.path.isdir(folder_path):
            continue
        
        for file_name in os.listdir(folder_path):
            count_0 = 0
            txt_data_list = []
            if not file_name.endswith(".txt"):
                continue
            file1_path = folder_path+"/"+file_name

            with open(file1_path,"r") as ftxt:
                datas = ftxt.read().split('\n')
            datas = [data for data in datas if data != '']

            for data in datas :
                tmp = [None] * 5
                tmp[0] = str(data.split(' ')[0])
                tmp[1] = float(data.split(' ')[1])
                tmp[2] = float(data.split(' ')[2])
                tmp[3] = float(data.split(' ')[3])
                tmp[4] = float(data.split(' ')[4])
                txt_data_list.append([tmp[0],tmp[1],tmp[2],tmp[3],tmp[4]])

            for i in range(len(txt_data_list)):
                if txt_data_list[i][0] =="0":
                    count_0 +=1

            if count_0 ==1:
                print("error : {}   player : {} ".format(file1_path,count_0))
                process_player(file1_path,folder_path,txt_data_list,count_0)




