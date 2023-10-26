import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import LoadData
import MLFC
import cv2
import os
import gc
import random

from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
    
    row = 128
    col = 256
    conv_block_list = [4, 8, 16]
    dense_block_list = [1024, 128, 16] 
       
    model = MLFC(row, col, conv_block_list, dense_block_list)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    history = model.fit([x1, x2, x3, x4, x5],
                        y_train_onehot,
                        epochs=100,
                        batch_size = 64,
                        callbacks=[callback],
                        validation_data=([v1, v2, v3, v4, v5], y_valid_onehot))
    
    
    y_pred = model.predict([t1, t2, t3, t4, t5])
    print(classification_report(y_test, list(np.argmax(y_pred, axis=1))))
  
    