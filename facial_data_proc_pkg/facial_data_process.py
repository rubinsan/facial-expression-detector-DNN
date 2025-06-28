"""
Script for loading and processing the facial emotion images dataset
"""

import sys
import numpy as np
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
import glob
import csv

# Global variables
emotion_ID = {'angry': 0, 'fear': 1, 'happy': 2,
              'neutral': 3, 'sad': 4, 'surprise': 5}
emotion_list = emotion_ID.keys()

def jpg_to_csv(type_of_data):
    """
    Convert the images in the dataset to CSV files for each emotion type.
    :param type_of_data: 'train' or 'test'
    """
    for emotion in emotion_list:
        filelist = glob.glob(f'./facial_emotion_detection_dataset/{type_of_data}/{emotion}/*.jpg')
        ID = emotion_ID[emotion]
        x_list = []
        csv_fname = type_of_data + '_' + emotion + '.csv'
        for fname in filelist:
            img_array = np.array(Image.open(fname).convert('L')).flatten()
            x = img_array.tolist()
            x.insert(0, ID) # Set the first element as the emotion ID
            x_list.append(x)
        with open(csv_fname, 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(x_list)
        del x_list 

def load_csv_data(type_of_data):
    """
    Load the CSV files for the specified type of data ('train' or 'test').
    :param type_of_data: 'train' or 'test'
    Returns:
        X_train -- train data matrix
        Y_train -- train labels (emotion IDs)
    """
    
    Raw_data = np.empty((1, 224*224+1), dtype=np.uint8) # Assuming 224*224 pix + ID
    for emotion in emotion_list:
        fname = './' + type_of_data + '_data/' + type_of_data + '_' + emotion + '.csv'
        X_emo = np.loadtxt(fname, delimiter=",", ndmin=2).astype(np.uint8)
        print(X_emo.shape)
        Raw_data = np.concatenate((Raw_data, X_emo), axis=0)

    print("Raw_data shape:", Raw_data.shape)
    np.random.shuffle(Raw_data)
    Raw_data = Raw_data.T # Once shuffled, transpose
    print(Raw_data[0:5, 0:10])
    print("Raw_data shape:", Raw_data.shape)
    Y_train = Raw_data[0, :]  # The first row is the emotion ID
    print("Y_train shape:", Y_train.shape)
    X_train = np.delete(Raw_data, 0, axis=0)
    print(X_train[0:5, 0:10])
    print("X_train shape:", X_train.shape)
    return X_train, Y_train

