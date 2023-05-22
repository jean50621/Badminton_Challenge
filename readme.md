# Badminton AI Coach


AICup URL:<https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8?lang=zhcupBadminton>


Our team participated in the competition "CoachAI Badminton Challenge" hold by AICup showing upon. The competition included several tasks realated in badminton. Here are the instructions and implementations for all of the tasks. Before using, players, net and court locations are required, which were detected by [yolov7](https://github.com/WongKinYiu/yolov7). We provided the weights that will be used in <https://idocntnu-my.sharepoint.com/:f:/g/personal/61175070h_eduad_ntnu_edu_tw/EnfiYWEL1zBPmJRxGZyw9r8BrPSu1GadgZaxgevkmAdPWQ?e=brvjxb>.


## Running Environment
see environment.txt

## Hardware
- [x] PC (CPU: i9-10900K, GPU: Nvidia Geforce RTX 3090)
- [x] Google Colab


## Inference

Some of the inputs might need the files generated by previous steps, make sure you get the input data correctly. All the requirments should be store in the folder 'data', including entire frames and the results from yolov7.

All the input information is in the YAML file (in each task folder).


### 01_ObjDetector

* We trained a YOLOv7 model to detect players, court, nets, and released the weights at {RELEASE_PRETRAINED_PATH}/01_ObjDetector/best.pt.

* download and run yolov7 (https://github.com/WongKinYiu/yolov7)

* If the players, court and net locations are noisy (e.g. one of  them are missing), you can run filter.py to remove them as following.
```
 python filter.py --c test.yaml
```

### 02_BallDetector
```
 python inference.py --c test.yaml
 python denoise.py --c test.yaml
```

### 03_HitFrame_Hitter
```
 python inference.py --c test.yaml
 python process_hitter_val.py --c test.yaml
 python transfer2csv.py --c test.yaml
 python split_csv.py --c test.yaml
```

### 04_PoseEstimator
For pose estimation, we use [RTMPose](https://github.com/open-mmlab/mmpose) to get the coordinate of players' feet. The installation is in the MMPose official site, please install all the requirments before starting the estimator. 
```
 python inference.py --c test.yaml
```

### 05_RoundHead
```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```

### 06_Backhand
```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```

### 07_BallHeight
```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```

### 08_BallLanding

```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```
In this step, we use [TrackNetv2](https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2) to process the badminton landing location of the last hit frame since our team can hardly predict it correctly. 
```
 python landing_inter.py
```

### 09_PlayerLocation
```
 python inference.py --c test.yaml
 python bind_two_results.py --c test.yaml
 python csv_writer.py --c test.yaml
```

### 10_BallType
```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```

### 11_Winner
```
 python inference.py --c test.yaml
 python csv_writer.py --c test.yaml
```

## Acknowledgement
