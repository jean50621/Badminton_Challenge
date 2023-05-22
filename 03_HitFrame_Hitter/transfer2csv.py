import pandas as pd
import os
import csv
from cfg import get_args

def read_txt(path):
    with open(path, "r") as ftxt:
        datas = ftxt.read().split('\n')
    datas = [data for data in datas if data != '']
    return datas

def save_csv(save_path, datas, video_name=None):

    with open(save_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if video_name is None:
            writer.writerow(datas)
        else:
            count = 1
            for data in datas:
                tmp = [None] * 15
                tmp[0] = str(video_name) + ".mp4"
                tmp[1] = count
                tmp[2] = int(float(data.split(' ')[1]))
                tmp[3] = data.split(' ')[0]
                tmp[4:] = [-1] * 11
                tmp[-1] = "X"
                writer.writerow(tmp)
                count += 1


if __name__ == "__main__":

    args = get_args()

    result_path = "../aicup_final_{}.csv".format(args.prefix)
    if os.path.isfile(result_path):
        os.remove(result_path)
    first_row = "VideoName ShotSeq HitFrame Hitter RoundHead Backhand BallHeight \
        LandingX LandingY HitterLocationX HitterLocationY DefenderLocationX \
        DefenderLocationY BallType Winner"
    first_row = first_row.split(' ')
    first_row = [x for x in first_row if x != '']
    save_csv(result_path, first_row)

    # os.makedirs(result_root, exist_ok=True)
    txt_path = "../results/result_hitters_frame_{}/".format(args.prefix)
    
    files = os.listdir(txt_path)
    for file in files:
        datas = read_txt(txt_path + "/" + file)
        video_name = file.split('.')[0].replace("{}_".format(args.prefix), "") # "val_"
        save_csv(result_path, datas, video_name)

