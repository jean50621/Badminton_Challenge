import csv
import os
from cfg import get_args

if __name__ == "__main__":
    args = get_args()
    rows = []
    open_path = "../aicup_final_{}.csv".format(args.prefix)
    with open(open_path, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            rows.append(row)

    checking_list = set([i for i in range(170, 400)])

    head = rows[0]
    head = head[1:]
    prvs_save_path = None
    datas = []

    if os.path.isfile("../val_csvs/"):
        os.remove("../val_csvs/")
    os.makedirs("../val_csvs/",exist_ok=True)
    for row in rows[1:]:
        save_path =  "../val_csvs/" + row[0].split('.')[0] + ".csv"
        
        if prvs_save_path is not None and save_path != prvs_save_path:
            if int(row[0].split('.')[0]) in checking_list:
                checking_list.remove(int(row[0].split('.')[0]))
            # save
            with open(prvs_save_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(head)
                for data in datas:
                    writer.writerow(data)
            datas = []

        datas.append(row[1:])
        prvs_save_path = save_path

    #print(checking_list)

    #print(save_path)
    # print(datas)
    if datas != []:
        with open(prvs_save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(head)
            for data in datas:
                writer.writerow(data)
        
        