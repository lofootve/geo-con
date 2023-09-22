import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import LoadData
import cv2
import os
import gc
import random

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    FILE_PATH = "WIRTE YOUR 3W FILE PATH"
    file_names = ['P-PDG.png', 'P-TPT.png', 'T-TPT.png', 'P-MON-CKP.png', 'T-JUS-CKP.png']
    
    dataloder = LoadData(FILE_PATH, file_names)
    data = dataloder.load_data(9)
    dataloder.save_image()
    
    x1, x2, x3, x4, x5, y_train, t1, t2, t3, t4, t5, y_test = dataloder.load_images()
    x1, v1, x2, v2, x3, v3, x4, v4, x5, v5, y_train, y_valid = train_test_split(x1, x2, x3, x4, x5, y_train, test_size=0.2, random_state=42)
    
    y_train_onehot = np.eye(9)[y_train]
    y_valid_onehot = np.eye(9)[y_valid]
    y_test_onehot = np.eye(9)[y_test]       
    
    
  
    