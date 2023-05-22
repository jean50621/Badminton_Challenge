import csv
import os
import random
from cfg import get_args
if __name__ == "__main__":
    args = get_args()
    result_root = args.save_root
    files = os.listdir(result_root)

    datas_to_write = {}
    for file in files:
        file_name = file.split('.')[0] + ".mp4"
        datas_to_write[file_name] = None
        with open(result_root + file, 'r') as ftxt:
            data = ftxt.read()
        datas_to_write[file_name] = data
        print(file_name, data)

    rows = []
    with open("../aicup_final_{}.csv".format(args.prefix), 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)


## for only one video

    for i in range(len(rows)):
        if i == 0:
            continue
        video_name = rows[i][0]
        rows[-1][-1] = datas_to_write[video_name]

        

## for multiple videos detection in csv

    # prvs_video_name = None
    # for i in range(len(rows)):
    #     if i == 0:
    #         continue
    #     video_name = rows[i][0]
    #     if prvs_video_name is not None and video_name != prvs_video_name:
    #         rows[i-1][-1] = datas_to_write[prvs_video_name]
    #     # rows[i][4:] = [-1] * 10 + ['X']
    #     elif i==len(rows)-1 :
    #         rows[i][-1] = datas_to_write[prvs_video_name]

    #     prvs_video_name = video_name

    with open("../aicup_final_{}.csv".format(args.prefix), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
