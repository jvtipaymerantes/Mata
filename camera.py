import cv2
import torch
import numpy as np


model = torch.hub.load(r"C:\Users\Bob\Downloads\yolov5\content\yolov5", 'custom', path=r"C:\Users\Bob\Downloads\yolov5\content\yolov5\runs\train\results_3\weights\best.pt", source='local', autoshape=True)
model.conf = 0.4  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.max_det = 1000  # maximum number of detections per image
model.amp = False
model.classes = 0

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
       
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        fps = int(self.video.get(cv2.CAP_PROP_FPS))
        image = cv2.flip(image, 1)
        results = model(image)
        a = np.squeeze(results.render())
        cv2.waitKey(fps)

        # encode the image to JPEG for displaying in Flask
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
