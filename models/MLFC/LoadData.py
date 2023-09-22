import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import cv2
import math
import gc
import random
from tqdm import tqdm
from glob import glob
from tqdm import tqdm

def getlim(data):
    col = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"]
    upper_bound = []
    lower_bound = []

    for name in col:
        l_1q = data[name].quantile(0.25)
        l_3q = data[name].quantile(0.75)
        IQR = l_3q - l_1q

        upper_bound.append(l_3q + (1.5 * IQR))
        lower_bound.append(l_1q - (1.5 * IQR))
    return upper_bound, lower_bound

def masking(image, left):
    for i in range(len(image)):
        for j in range(left, left+2):
            for k in range(0, 3):
                image[i][j][k] = 0

    return image

def match_size(i, image):
  if i == 0:
    image = image[:, :-5]
  elif i == 1:
    image = image[:, :-1]
  elif i == 2:
    image = image[:, :-2]
  elif i == 3:
    image = image[:, :-3]
  elif i == 4:
    image = image[:, :-4]
  return image

def append_data(x1, x2, x3, x4, x5, data):
    x1 = np.append(x1, data[0][np.newaxis,:,:,:], axis=0)
    x2 = np.append(x2, data[1][np.newaxis,:,:,:], axis=0)
    x3 = np.append(x3, data[2][np.newaxis,:,:,:], axis=0)
    x4 = np.append(x4, data[3][np.newaxis,:,:,:], axis=0)
    x5 = np.append(x5, data[4][np.newaxis,:,:,:], axis=0)

    return x1, x2, x3, x4, x5

class LoadData:
    def __init__(self, path, file_names, image_path="./results"):
        self.path = path
        self.file_names = file_names
        self.image_path = image_path
        
    def load_images(self):
        test_index = [(4, 317), (0, 553), (0, 180), (0, 100), (4, 97), (0, 83), (0, 256), (4, 199), (4, 149), (0, 293), (7, 1), (0, 173), (4, 142), (3, 104), (0, 476), (0, 51), (4, 340), (0, 135), (0, 581), (4, 65), (4, 94), (4, 310), (4, 268), (5, 2), (0, 153), (0, 481), (0, 288), (0, 508), (4, 321), (4, 279), (0, 347), (0, 44), (4, 264), (0, 379), (4, 309), (0, 307), (0, 196), (0, 58), (0, 361), (0, 162), (4, 30), (4, 261), (0, 569), (0, 354), (4, 138), (4, 125), (4, 181), (4, 50), (0, 545), (0, 509), (0, 446), (0, 132), (4, 329), (4, 251), (4, 34), (0, 262), (4, 175), (0, 584), (0, 165), (3, 81), (0, 200), (0, 225), (4, 107), (0, 112), (0, 203), (0, 501), (0, 91), (0, 389), (0, 442), (0, 280), (3, 99), (0, 522), (4, 263), (0, 70), (0, 142), (0, 496), (0, 488), (0, 351), (3, 92), (0, 417), (0, 194), (4, 334), (0, 42), (4, 319), (0, 47), (4, 75), (4, 45), (0, 167), (4, 267), (0, 195), (0, 230), (0, 406), (0, 549), (0, 409), (3, 101), (4, 46), (0, 136), (4, 203), (0, 188), (0, 516), (0, 461), (7, 2), (0, 177), (0, 325), (0, 339), (0, 531), (0, 539), (4, 24), (4, 71), (0, 396), (5, 3), (0, 106), (0, 474), (0, 220), (0, 449), (0, 198), (4, 111), (0, 378), (0, 515), (0, 538), (0, 331), (3, 102), (0, 350), (0, 557), (4, 14), (4, 28), (0, 81), (0, 513), (0, 123), (4, 57), (4, 275), (0, 59), (0, 107), (3, 94), (4, 54), (4, 63), (0, 445), (0, 20), (4, 82), (0, 407), (0, 197), (3, 78), (2, 8), (4, 47), (3, 93), (0, 1), (0, 344), (0, 286), (0, 342), (0, 517), (0, 268), (0, 85), (0, 413), (4, 298), (3, 98), (4, 96), (4, 170), (4, 316), (4, 16), (4, 258), (4, 259), (4, 58), (0, 206), (0, 591), (2, 10), (4, 236), (0, 147), (0, 102), (4, 322), (4, 225), (4, 127), (4, 20), (4, 38), (4, 61), (0, 226), (0, 500), (4, 64), (4, 201), (0, 19), (0, 248), (0, 284), (0, 212), (0, 561), (0, 61), (0, 146), (0, 422), (0, 343), (0, 468), (0, 222), (0, 170), (4, 231), (0, 260), (0, 579), (0, 370), (0, 34), (0, 497), (4, 202), (3, 76), (0, 401), (0, 455)]
        dic_list = os.listdir(self.image_path)
        dic_list.sort()

        upscale_dic = {1:100, 2:10, 5:10, 6:100, 7:100}

        x1 = np.empty([0,128,251,3])
        x2 = np.empty([0,128,255,3])
        x3 = np.empty([0,128,254,3])
        x4 = np.empty([0,128,253,3])
        x5 = np.empty([0,128,252,3])
        y_train = []

        t1 = np.empty([0,128,251,3])
        t2 = np.empty([0,128,255,3])
        t3 = np.empty([0,128,254,3])
        t4 = np.empty([0,128,253,3])
        t5 = np.empty([0,128,252,3])
        y_test = []

        images = []
        count = 0
        class_num = 0
        class_temp = []
        for i in tqdm(dic_list):
            temp = []

            check_index = (i.split("_"))
            check_index = (int(check_index[0]), int(check_index[1]))
            if check_index in test_index:
                for j, name in enumerate(self.file_names):
                    t_temp = cv2.imread(self.image_path + "/{}/{}".format(i, name))
                    t_temp = cv2.resize(t_temp, (256, 128))
                    t_temp = (255.0 - t_temp) / 255.0
                    t_temp = match_size(j, t_temp)
                temp.append(t_temp)
                t1, t2, t3, t4, t5 = append_data(t1, t2, t3, t4, t5, temp)
                y_test.append(check_index[0])

            else:
                for j, name in enumerate(self.file_names):
                    t_temp = cv2.imread(self.image_path + "/{}/{}".format(i, name))
                    t_temp = cv2.resize(t_temp, (256, 128))
                    t_temp = (255.0 - t_temp) / 255.0
                    t_temp = match_size(j, t_temp)
                temp.append(t_temp)

                #1, 2, 5, 6, 7번 이상 사건에 대해서는 임의 추출한 2개의 컬럼에서 임의의 3%데이터를 마스킹하여 upscaling
                if check_index[0] == 1 or check_index[0] == 2 or check_index[0] == 5 or check_index[0] == 6 or check_index[0] == 7 :
                    for i in range(upscale_dic[check_index[0]]):
                        t_temp = temp
                        rand = random.sample(range(5), 2)
                        for j in rand:
                            for k in range(3):
                                left = random.randint(0, 249)
                                t_temp[j] = masking(t_temp[j], left)

                        x1 = np.append(x1, t_temp[0][np.newaxis,:,:,:], axis=0)
                        x2 = np.append(x2, t_temp[1][np.newaxis,:,:,:], axis=0)
                        x3 = np.append(x3, t_temp[2][np.newaxis,:,:,:], axis=0)
                        x4 = np.append(x4, t_temp[3][np.newaxis,:,:,:], axis=0)
                        x5 = np.append(x5, t_temp[4][np.newaxis,:,:,:], axis=0)
                        y_train.append(check_index[0])
                else:
                    x1 = np.append(x1, temp[0][np.newaxis,:,:,:], axis=0)
                    x2 = np.append(x2, temp[1][np.newaxis,:,:,:], axis=0)
                    x3 = np.append(x3, temp[2][np.newaxis,:,:,:], axis=0)
                    x4 = np.append(x4, temp[3][np.newaxis,:,:,:], axis=0)
                    x5 = np.append(x5, temp[4][np.newaxis,:,:,:], axis=0)
                    y_train.append(check_index[0])
        return x1, x2, x3, x4, x5, y_train, t1, t2, t3, t4, t5, y_test

    def load_data(self, num):
        #num : number of annotation class
        for event in range(num):
            file_lst = []
            full_path = self.path + f'{event}/*.csv'
            file_lst.extend(sorted(glob(full_path)))
            globals()['df_event_{}'.format(event)] = di.dt_integration(file_lst)
            globals()['df_event_{}'.format(event)].drop(columns = ['timestamp', 'event_type', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL', 'instance_type'], inplace = True)
            globals()['df_event_{}'.format(event)].reset_index(inplace=True, drop = True)
            
            ###############
            ###############
            
    def save_image(self):
        df = []
        df_up = []
        df_low = []
        for i in range(9):
            data = pd.read_csv(self.path + "/merged_df_{}.csv".format(i))
            data = data.drop(["P-JUS-CKGL", "T-JUS-CKGL", "QGL"], axis=1) #칼럽삭제
            if i == 1 or i == 2 or i == 5 or i == 6 or i == 7 :
                data = data[data["instance_type"] == "WELL"] #시뮬레이티드, 그린 인스턴스 삭제
            else:
                data = data[(data["instance_type"] == "WELL") | (data["instance_type"] == "SIMULATED")] #그린 인스턴스 삭제
            data = data.reset_index(drop=True)
            temp_up, temp_low = getlim(data)
            df_up.append(temp_up)
            df_low.append(temp_low)
            file_name = data["file_name"][0]
            prev = 0

            temp = []
            for j in range(len(data)):
                if file_name != data["file_name"][j]: #각 인스턴스별 분리
                    temp.append(data.loc[prev:j-1])
                    file_name = data["file_name"][j]
                    prev = j
            df.append(temp)
            del temp
        del data
        
        columns = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"] 
  
        for i in tqdm(range(9)):
            for j in range(len(data[i])):
                for k in range(len(columns)):
                    fig = plt.figure(frameon=False)
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    plt.plot(data[i][j][columns[k]])
                    plt.ylim([df_low[i][k], df_up[i][k]])
                    plt.savefig(self.image_path + "/{}_{}/{}.png".format(i, j, columns[k]), bbox_inches="tight", pad_inches = 0)
                    plt.clf()
                    plt.close()
                    
            
        
