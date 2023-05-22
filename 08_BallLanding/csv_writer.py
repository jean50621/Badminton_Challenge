import csv
import os
import random
from cfg import get_args

if __name__ == "__main__":

    args = get_args()
    files = os.listdir(args.result_root)
    
    datas_to_write = {}
    for file in files:
        file_name = file.split('.')[0] + ".mp4"
        datas_to_write[file_name] = []
        with open(args.result_root + file, 'r') as ftxt:
            datas = ftxt.read().split('\n')
        for data in datas:
            if data == '':
                continue
            infos = data.split(' ')
            datas_to_write[file_name].append([int(round(float(infos[3]))), 
                                              int(round(float(infos[4])))])

    rows = []
    with open("../aicup_final_val.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    video_data_count = {}
    prvs_video_name = None
    count = 0
    for i in range(len(rows)):
        if i == 0:
            continue
        video_name = rows[i][0]
        # print(video_name, video_name in datas_to_write)
        assert video_name in datas_to_write
        if video_name not in video_data_count:
            video_data_count[video_name] = 0
        if video_data_count[video_name] == len(datas_to_write[video_name]):
            print(count + 1)
            count += 1
            continue
        _res = datas_to_write[video_name][video_data_count[video_name]]
        video_data_count[video_name] += 1
        rows[i][7] = _res[0]
        rows[i][8] = _res[1]

    with open("../aicup_final_val.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
