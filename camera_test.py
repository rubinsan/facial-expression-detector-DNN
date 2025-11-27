import torch
from mtcnn import MTCNN
import pyrealsense2 as rs
import numpy as np
import cv2
from main_CNN_VGG19 import NeuralNetwork, device

# Initialize MTCNN detector
mtcnn = MTCNN(device="GPU:0")

# Create facial expression detector model and load weights
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model_VGG19.pth', weights_only=True))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Detect faces and landmarks
        bounding_boxes = mtcnn.detect_faces(color_image, threshold_onet=0.85)

        # Show images
        for box in bounding_boxes:
            x1, y1, w, h = box['box']
            cv2.rectangle(color_image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()