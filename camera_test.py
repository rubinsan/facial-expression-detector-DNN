import torch
from mtcnn import MTCNN
import pyrealsense2 as rs
import numpy as np
import cv2
from CNN_models import VGG, RESNET

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Initialize MTCNN detector
mtcnn = MTCNN(device="GPU:0")

# Create facial expression detector model and load weights
# Choose option VGG or ResNet architecture
option = "VGG"
#option = "ResNet"

if option == "VGG": 
    model = VGG.VGG19().to(device)
    model.load_state_dict(torch.load('CNN_models/Weights/model_VGG19.pth', weights_only=True))
elif option == "ResNet": 
    model = RESNET.ResNet(RESNET.BasicBlock, [2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load('CNN_models/Weights/model_RESNET18.pth', weights_only=True))

model.eval()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

face_model_input_size = (48,48)

def emo_class(emo_ID):
    match emo_ID:
        case 0:
            return "angry"
        case 1:
            return "disgust"
        case 2:
            return "fear"
        case 3:
            return "happy"
        case 4:
            return "neutral"
        case 5:
            return "sad"
        case 6:
            return "surprise"

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        resized_gray = np.zeros((face_model_input_size))

        # Detect faces and landmarks
        bounding_boxes = mtcnn.detect_faces(color_image, threshold_onet=0.85)

        # Crop face and preprocess to fit expression detector model input format
        for box in bounding_boxes:
            x1, y1, w, h = box['box']
            cv2.rectangle(color_image, (x1, y1), (x1+w,y1+h), (0,255,0), 2)
            cropped_img = color_image[y1:y1+h, x1:x1+w]	
            resized_img = cv2.resize(cropped_img, face_model_input_size)
            resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            input_data = resized_gray.astype(np.float32) / 255
            input_data = np.expand_dims(input_data, axis=0)
            with torch.no_grad():
                X = torch.from_numpy(input_data).to(device)
                pred = model(X.unsqueeze(0))
                emotion = pred.argmax(1)
                emotion = emo_class(emotion.item())
            image = cv2.putText(color_image, emotion, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.namedWindow('Emotion Detector', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('Copped face', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Emotion Detector', color_image)
        #cv2.imshow('Copped face', resized_gray)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()