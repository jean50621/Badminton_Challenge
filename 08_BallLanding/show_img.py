import cv2
import os

if __name__ == "__main__":
    
    image_root = "D:/D-Projects-23/AICUP23/DATA/entir_frames/"
    
    save_root = "./landings/"
    
    results_root = './results/'
    for i in range(170, 190):

        result_txt = str(i).zfill(5) + ".txt"
        with open(results_root + result_txt, "r") as ftxt:
            preds = ftxt.read().split('\n')

        video_name = "test_" + str(i).zfill(5)
        os.makedirs(save_root + video_name + "/", exist_ok=True)
        
        for pred in preds:
            if pred == '':
                continue

            ps = pred.split(' ')
            idx = ps[0]
            frame_id = idx + ".jpg"
            
            px, py = float(ps[1]), float(ps[2])
            px, py = int(px), int(py)
            
            x, y = float(ps[3]), float(ps[4])
            x, y = int(x), int(y)

            arr = cv2.imread(image_root + video_name + "/" + frame_id)

            cv2.circle(arr, (px, py), 2, (255, 0, 255), -1)
            cv2.circle(arr, (x, y), 2, (0, 0, 255), -1)

            if ps[5] != "None":
                x_b = float(ps[5])
                x_b = int(x_b)
                y_b0 = 10
                y_b1 = y + 10

                cv2.line(arr, (x_b, y_b0), (x_b, y_b1), (0, 0, 255), 1)

                cv2.putText(arr, str(abs(x_b - x)), (30, 30), 1, 2, (0, 0, 255), 1, 1)

            cv2.imwrite(save_root + video_name + "/" + frame_id, arr)
            
            # cv2.line(arr, (x, y-200), (x, y+200), (0, 0, 255), 1)
            # cv2.namedWindow('frame', 0)
            # cv2.imshow('frame', arr)
            # cv2.waitKey(0)