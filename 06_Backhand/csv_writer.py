import csv
import os
from cfg import get_args
if __name__ == "__main__":
    args = get_args()

    result_root = args.save_root
    files = os.listdir(result_root)

    datas_to_write = {}
    for file in files:
        file_name = file.split('.')[0] + ".mp4"
        datas_to_write[file_name] = []
        with open(result_root + file, 'r') as ftxt:
            datas = ftxt.read().split('\n')
        for data in datas:
            if data == '':
                continue
            datas_to_write[file_name].append(int(data.split(' ')[0]))

    rows = []
    with open(args.csv_root, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)


    video_data_count = {}
    for i in range(len(rows)):
        if i == 0:
            continue
        video_name = rows[i][0]
        print(video_name, video_name in datas_to_write)
        assert video_name in datas_to_write
        if video_name not in video_data_count:
            video_data_count[video_name] = 0
        _res = datas_to_write[video_name][video_data_count[video_name]]
        video_data_count[video_name] += 1
        rows[i][5] = _res

    with open(args.csv_root, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)
