import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def preprocess(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    ret, ret_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #apply binary threshold
    #cv2.fastNlMeansDenoising(img, img)
    resz_img = cv2.resize(ret_img, (128,128)) #resize it to 25*25 image
    
    return resz_img
        
def load_data():
    cur_path = os.getcwd()
    pre_path = os.path.join(os.getcwd(), 'preprocess')
    x_train = []
    y_train = []

    # path to pre-processed images
    folder_list = os.listdir(cur_path + "/data/")
    os.chdir(cur_path + "/data/")

    print(folder_list)

    for folder_name in folder_list:
        y_label = folder_name.split("-")[0]
        
        #print("y_label is " + y_label)

        if (y_label[0] == "."):
            continue

        filelist = os.listdir(folder_name)
        os.chdir(folder_name)

        for ifile in filelist:
            if (ifile.endswith(".jpg") and not ifile.startswith(".")):
                
                img = cv2.imread(ifile,0)   
                img = preprocess(img)
                cv2.imwrite(os.path.join(pre_path, ifile), img)
                x_train.append(img)
                y_vec = label2vec (y_label)
                y_train.append(y_vec)
                
        
        os.chdir(cur_path + "/data/")
        
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print (x_train.shape, y_train.shape)
    
    return x_train, y_train

def vec2int(y_vec):
    y_decoded = []
    for x in range(0,len(y_vec)):
        if y_vec[x] != 0:
            y_decoded.append(int(x+2304))   
    
    return y_decoded    


def label2vec (y_label):
    y_vec = np.zeros(8)
    
    if y_label == "bed":
        y_vec[0] = 1
    if y_label == "chair":
        y_vec[1] = 1
    if y_label == "lamp":
        y_vec[2] = 1
    if y_label == "shelf":
        y_vec[3] = 1
    if y_label == "sofa":
        y_vec[4] = 1
    if y_label == "stool":
        y_vec[5] = 1
    if y_label == "table":
        y_vec[6] = 1
    if y_label == "wardrobe":
        y_vec[7] = 1

    return y_vec

if __name__=='__main__':
    
    x_train, y_train = load_data()
    print x_train.shape, y_train.shape
    print x_train[0].shape
    plt.imshow(x_train[0])
    plt.show()
