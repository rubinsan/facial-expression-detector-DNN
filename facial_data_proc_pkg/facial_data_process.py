"""
Script for loading and processing the facial emotion images dataset
"""

import sys
import numpy as np
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
import glob
import csv
import itertools

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

def load_csv_data(type_of_data, n_samples=1000):
    """
    Load the CSV files for the specified type of data ('train' or 'test').
    :param type_of_data: 'train' or 'test'
    Returns:
        X -- data matrix
        Y -- labels according to emotion IDs
    """
    if type_of_data not in ['train', 'test']:
        raise ValueError("type_of_data must be 'train' or 'test'")
    
    print(f"Start loading {type_of_data}ing data...")
    Raw_data = np.empty((1, 224*224+1), dtype=np.uint8) # 224*224 pix + ID
    for emotion in emotion_list:
        fname = './' + type_of_data + '_data/' + type_of_data + '_' + emotion + '.csv'
        with open(fname) as f:
            X_emo = np.genfromtxt(itertools.islice(f,0,n_samples),
                                  delimiter=',').astype(np.uint8)

        #X_emo = np.loadtxt(fname, delimiter=",", ndmin=2).astype(np.uint8)
        Raw_data = np.concatenate((Raw_data, X_emo), axis=0)
        print("Loading...")

    Raw_data = np.delete(Raw_data, 0, axis=0)  # Remove the first garbage row
    np.random.shuffle(Raw_data)
    Raw_data = Raw_data.T # Once shuffled, transpose
    #print("Raw_data shape after transpose:", Raw_data.shape)
    #print(Raw_data[0:5, 0:10])
    Y = Raw_data[0, :]  # The first row is the emotion ID
    #print("Y shape:", Y.shape)

    Y_onehot = np.zeros((Y.max() + 1, Y.size), dtype=np.int8)
    Y_onehot[Y, np.arange(Y.size)] = 1
    #print("Y_onehot shape:", Y_onehot.shape)
    #print(Y_onehot[:, 0:10])

    X = np.delete(Raw_data, 0, axis=0) / 255.0 # Normalize pixel values to [0,1]
    #print(X[0:5, 0:10])
    #print("X shape:", X.shape)
    print(f"{type_of_data} data loaded successfully!")
    return X, Y_onehot
