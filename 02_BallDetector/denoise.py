import os
import cv2
import numpy as np
from cfg import get_args


##haha
def get_files(root: str) -> dict:
    res = {}
    files = os.listdir(root)
    files_set = set(files)
    for file in files:
        ext = file.split('.')[-1]
        name = file.split('.')[0]
        if ext != "mp4":
            continue
        if (name + ".txt" in files_set) and (name + ".mp4" in files_set):
            res[name] = 1
    return res

class BallLocFilter(object):

    def __init__(self, 
                 video_path: str, 
                 label_path: str, 
                 score_threshold: float,
                 save_path: str = None):
        """
        if 'save_path' is None: would not save video
        """
        self.video_path = video_path
        self.label_path = label_path
        self.cap = cv2.VideoCapture(self.video_path)
        self.labels = self.load_txt(self.label_path)
        self.score_threshold = score_threshold
        self.save_path = save_path # result video's saving path
        self.video_writer = None # create at stream() function has been called

        self.locs = []
        self.loc_frames = []

    def load_txt(self, txt_path: str):
        with open(txt_path, "r") as ftxt:
            datas = ftxt.read().split('\n')
        datas = [x for x in datas if x != '']
        return datas

    def plot_bbox(self, frame, x, y, score):
        max_col = int(round(float(x)))
        max_row = int(round(float(y)))
        cv2.rectangle(frame, 
                     (max_col-3, max_row-3), 
                     (max_col+3, max_row+3), 
                     (0, 0, 255), 1)
        score_txt = "{}".format(round(float(score), 4))
        cv2.putText(frame, score_txt, (max_col-4, max_row-3), 1, 1, (0, 0, 255), 1, 1)

    def show_frame(self, frame, name='frame', wait_time=33):
        cv2.nemadWindow(name, 0)
        cv2.imshow(name, frame)
        cv2.waitKey(wait_time)

    def stream_next(self, frame_id):
        ret, frame = self.cap.read()
        if not ret: 
            return 0
        ball_loc = self.labels[frame_id]
        y, x, score = ball_loc.split(' ')
        y, x, score = float(y), float(x), float(score)
        y = y * (960 / 1280) * (720 / 520)
        x = x * (520 / 720) * (1280 / 960)
        # === score ===
        if not self.filter_by_score(score):
            return 1

        # === record data ===
        self.locs.append(np.array([x, y]))
        self.loc_frames.append(frame_id)

        # plot and write result
        self.plot_bbox(frame, x, y, score)
        return 2

    def l2loss(self, loc1, loc2):
        err = loc1 - loc2
        err = (err[0] * err[0] + err[1] * err[1]) ** 0.5
        return err

    def stream(self):
        frame_id = 0
        while True:
            sat = self.stream_next(frame_id)
            if sat == 0:
                break
            frame_id += 1 # next frame id
        self.cap.release()

        block_frame_ids = []
        vecs = self.build_vecs()
        block_frame_ids += self.filter_forward_line(vecs)
        vecs_b = self.build_vecs_back()
        block_frame_ids += self.filter_backward_line(vecs_b)
        block_frame_ids += self.filter_too_far()
        block_frame_ids += self.filter_border()
        block_frame_ids  = set(block_frame_ids)

        # re-save
        msg = ""
        if self.save_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.save_path + ".mp4", fourcc, 30.0, 
                (1280,  720), isColor=1)
        self.cap = cv2.VideoCapture(self.video_path)
        frame_id = -1
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1
            use_frame = 1
            cv2.putText(frame, str(frame_id), (10, 10), 1, 1, (0, 0, 255), 1, 1)

            if (frame_id == 0) or (frame_id in block_frame_ids):
                use_frame = 0

            ball_loc = self.labels[frame_id]
            y, x, score = ball_loc.split(' ')
            y, x, score = float(y), float(x), float(score)
            y = y * (960 / 1280) * (720 / 520)
            x = x * (520 / 720) * (1280 / 960)
            # === score ===
            if not self.filter_by_score(score):
                use_frame = 0

            if use_frame == 1:
                self.plot_bbox(frame, x, y, score)
            self.video_writer.write(frame)
            
            x = int(round(float(x)))
            y = int(round(float(y)))
            msg += "{} {} {} {} {}\n".format(frame_id, x, y, score, use_frame)

        self.video_writer.release()

        with open(self.save_path + ".txt", "w") as ftxt:
            ftxt.write(msg)

    def build_vecs(self):
        vecs = []
        for i in range(len(self.locs) - 1):
            vec = self.locs[i+1] - self.locs[i]
            vecs.append(vec)

        for i in range(1, len(vecs)):
            if vecs[i][0] == 0 and vecs[i][1] == 0:
                vecs[i] = vecs[i-1]
        return vecs

    def build_vecs_back(self):
        vecs_b = []
        for i in range(len(self.locs) - 1, 0, -1):
            vec = self.locs[i-1] - self.locs[i]
            vecs_b.append(vec)

        for i in range(1, len(vecs_b)):
            if vecs_b[i][0] == 0 and vecs_b[i][1] == 0:
                vecs_b[i] = vecs_b[i-1]
        return vecs_b


    """ 
    following: filter functions 
    filter function rule:
        Return False if not pass the condition.
        That means the ball here is a noise point.
    """
    def filter_by_score(self, score):
        if score < self.score_threshold:
            return False
        return True

    def filter_forward_line(self, vecs) -> list:
        errs = []
        noisy = []
        pred = None
        for i in range(1, len(self.locs)):
            vi = i - 1
            dvec = vecs[vi]
            if pred is not None:
                err = self.l2loss(pred, self.locs[i])
                if err > 50 and i < len(self.locs) - 1:
                    for j in range(1, 4):
                        if i+j > len(self.locs) - 1:
                            break
                        err2 = self.l2loss(pred, self.locs[i+j])
                        if err2 < 50:
                            # i loc is noise
                            noisy.append(self.loc_frames[i])
                            dvec = self.locs[i+j] - pred
                            break
                errs.append(err)
            # predict step
            pred = self.locs[i] + dvec

        return noisy

    def filter_backward_line(self, vecs_b) -> list:
        errs_b = []
        noisy_b = []
        pred = None
        for i in range(len(self.locs) - 2, 0, -1):
            vi = len(self.locs) - 2 - i
            dvec = vecs_b[vi]
            if pred is not None:
                err = self.l2loss(pred, self.locs[i])
                if err > 50 and i > 1:
                    for j in range(1, 4):
                        if i-j < 0:
                            break
                        err2 = self.l2loss(pred, self.locs[i-j])
                        if err2 < 50:
                            # i loc is noise
                            noisy_b.append(self.loc_frames[i])
                            dvec = self.locs[i-j] - pred
                            break
                errs_b.append(err)
            # predict step
            pred = self.locs[i] + dvec
        return noisy_b

    def filter_too_far(self):
        noisy3 = []
        for i in range(1, len(self.locs)-1):
            if (self.l2loss(self.locs[i-1], self.locs[i]) > 200) and (self.l2loss(self.locs[i+1], self.locs[i]) > 200):
                noisy3.append(self.loc_frames[i])
            if self.l2loss(self.locs[i-1], self.locs[i]) > 300:
                noisy3.append(self.loc_frames[i])
            if self.l2loss(self.locs[i+1], self.locs[i]) > 300:
                noisy3.append(self.loc_frames[i])
        return noisy3

    def filter_border(self):
        noisy4 = []
        for i in range(0, len(self.locs)):
            if self.locs[i-1][0] < 100 or self.locs[i-1][0] > 1180:
                noisy4.append(self.loc_frames[i])
        return noisy4

if __name__ == "__main__":

    args = get_args()
    files = get_files(args.datas_root)
    os.mkdir(args.denoise_save_root)
    
    for file in files:
        print("process...", file)
        bf = BallLocFilter(video_path = args.datas_root + file + ".mp4", 
                           label_path = args.datas_root + file + ".txt", 
                           score_threshold = 0.985,
                           save_path = args.denoise_save_root + file)
        bf.stream()