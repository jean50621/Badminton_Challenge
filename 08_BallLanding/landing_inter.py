import csv
import os

def read_pred(root: str) -> dict:
	pred_files = os.listdir(root)
	datas = {}
	for pred_file in pred_files:
		if pred_file.split('.')[1] == 'csv':
			video_name = pred_file.split('.')[0].split("_")[1]
			datas[video_name] = []
			with open(root+pred_file, 'r') as f:
				csvreader = csv.reader(f)
				for idx, row in enumerate(csvreader):
					if idx == 0:
						continue
					datas[video_name].append(row)
	
	return datas
					
def adjust_last(org_data: list, datas: dict) -> dict:
	landing = {}
	for i in range(len(org_data)):
		if i == 0:
			continue
		video_name = org_data[i][0].split('.')[0]
		frame = org_data[i][2]
		x_count = 0
		y_count = 0
		if org_data[i][7] == '-1':
			# print(video_name)
			# print(frame)
			count = 0

			#get the last five frame location if the visibility is 1 and get average
			for k in range(len(datas[video_name])):
				last_idx = len(datas[video_name])-1-k
				if datas[video_name][last_idx][1] == '1':
					x_count += int(datas[video_name][last_idx][2])
					y_count += int(datas[video_name][last_idx][3])
					count += 1
					if count == 5:
						break
			
			landing[video_name] = [frame, x_count*0.2, y_count*0.2]
	# print(landing)
	return landing




if __name__ == '__main__':
	# read org csv and save in list
	org_data = []
	with open("../aicup_final_val.csv", 'r') as f:
		csvreader = csv.reader(f)
		for row in csvreader:
			org_data.append(row)

	# read prediction csv
	pred_root = './tracknet_prediction/'
	datas = read_pred(pred_root)

	# change -1
	landing = adjust_last(org_data, datas)

	for i in range(len(org_data)):
		if i == 0:
			continue
		video_name = org_data[i][0].split('.')[0]
		if org_data[i][2] == landing[video_name][0]:
			org_data[i][7] = int(landing[video_name][1])
			org_data[i][8] = int(landing[video_name][2])
	
	# write into csv
	with open('../aicup_final_val.csv', 'w') as fcsv:
		writer = csv.writer(fcsv)
		for row in org_data:
			writer.writerow(row)
		





