import cv2
import torch
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from TennisProject.tracknet import BallTrackerNet

class BallDetector:
    def __init__(self, path_model, device='cuda'):
        self.model = BallTrackerNet(input_channels=9, out_channels=256)
        self.device = device
        self.model.load_state_dict(torch.load(path_model, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.width = 640
        self.height = 360

    def infer_model(self, frames):
        ball_track = [(None, None)] * 2
        prev_pred = [None, None]
        for num in tqdm(range(2, len(frames)), desc='Ball Detection'):
            img = cv2.resize(frames[num], (self.width, self.height))
            img_prev = cv2.resize(frames[num-1], (self.width, self.height))
            img_preprev = cv2.resize(frames[num-2], (self.width, self.height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            with torch.no_grad():
                out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, prev_pred)
            prev_pred = [x_pred, y_pred]
            ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=2, max_dist=80):
        feature_map *= 255
        feature_map = feature_map.reshape((self.height, self.width))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                                   param1=50, param2=2, minRadius=2, maxRadius=7)
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0] * scale
                    y_temp = circles[0][i][1] * scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break
            else:
                x = circles[0][0][0] * scale
                y = circles[0][0][1] * scale
        return x, y

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps

def visualize_and_save(frames, ball_track, fps, output_path):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, frame in enumerate(tqdm(frames, desc='Writing Output')):
        img_res = frame.copy()
        if i < len(ball_track) and ball_track[i][0] is not None:
            x, y = int(ball_track[i][0]), int(ball_track[i][1])
            img_res = cv2.circle(img_res, (x, y), radius=5, color=(0, 255, 0), thickness=2)
            img_res = cv2.putText(img_res, 'ball', org=(x + 8, y + 8),
                                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                                  thickness=2, color=(0, 255, 0))
        out.write(img_res)
    out.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, required=True)
    parser.add_argument('--path_input_video', type=str, required=True)
    parser.add_argument('--path_output_video', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print('Loading video...')
    frames, fps = read_video(args.path_input_video)
    print(f'Loaded {len(frames)} frames at {fps} fps')

    print('Running ball detection...')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)

    print('Generating output video...')
    visualize_and_save(frames, ball_track, fps, args.path_output_video)
    print(f'Done! Output saved to {args.path_output_video}')
