import os
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

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

class IClassify:
    
    def __init__(self):
        self.learning_rate = 0.001
        self.data_path = os.path.join(os.getcwd(), 'data')
        self.save_path = os.path.join(os.getcwd(), 'save')
        self.class_encoding = {0:"bed", 1: "chair", 2:"lamp", 3:"shelf",4:"sofa", 5:"stool", 6:"table", 7:"wardrobe"}
        self.model = None
    
    def load_data(self):
        cur_path = os.getcwd()
    
        x_train = []
        y_train = []

        # path to pre-processed images
        folder_list = os.listdir(cur_path + "/data/")
        os.chdir(cur_path + "/data/")

        for folder_name in folder_list:
            y_label = folder_name.split("_")[0]
            
            #print("y_label is " + y_label)

            if (y_label[0] == "."):
                continue

            filelist = os.listdir(folder_name)
            os.chdir(folder_name)

            for ifile in filelist:
                if (ifile.endswith(".jpg") and not ifile.startswith(".")):
                    
                    img = cv2.imread(ifile,0) 
                    img = self.preprocess(img)
                    x_train.append(img.reshape(64,64,1))
                    y_vec = label2vec (y_label)
                    y_train.append(y_vec)
            
            os.chdir(cur_path + "/data/")
            
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print (x_train.shape, y_train.shape)
        
        return x_train, y_train
        
    def preprocess(self, img):
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        ret, ret_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #apply binary threshold
        #cv2.fastNlMeansDenoising(img, img)
        resz_img = cv2.resize(ret_img, (64,64)) #resize it to 25*25 image
        
        return resz_img
        
    def make_model(self):
        self.model = Sequential()
        
        self.model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1),
                 activation='relu', input_shape=(64,64,1)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(512, (3, 3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate), metrics=[categorical_accuracy])

    def train(self):
        X,Y = self.load_data() #loads train data, preprocess it and return in desired format
        
        X = X.astype('float32')
        
        X, Y = shuffle(X, Y, random_state=0)
        
        train_datagen = ImageDataGenerator(
            featurewise_center = False,
            featurewise_std_normalization = False,
            rotation_range=10,
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest')
                
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size=0.5)
        
        X_test = X_test/255.
        X_validate = X_validate/255.
        
        train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
        
        if os.path.isfile(self.save_path + '/my_model.h5'):
            self.model = load_model(self.save_path + '/my_model.h5') 
        else:
            self.make_model()
        
        self.model.fit_generator(
                    train_generator,
                    steps_per_epoch=len(X_train)/32,
                    epochs=25,
                    validation_data=(X_validate, Y_validate),
                    validation_steps=len(X_validate)/32)
                    
        score,acc = self.model.evaluate(X_test, Y_test)
        self.model.save(os.path.join(self.save_path, 'my_model.h5'))
        print 'score: ', score, 'accuraccy: ', acc
        
    def predict(self, img):
        #img = cv2.imread(filepath, 0)
        img = self.preprocess(img)
        img1 = img.astype('float32')
        img1 = img1/255.0
        img1 = img1.reshape((1,64,64,1))
        preds = self.model.predict(img1)
        pred = np.argmax(preds[0])
        
        ret = np.array([0]*8)
        ret[pred] = 1
        print self.class_encoding[pred]
        
        return ret
        
if __name__ == "__main__":
    iclass = IClassify()
    #iclass.train()
    
    iclass.model = load_model(iclass.save_path + '/2015A7PS0115G.h5')
    
    # Read images and call predict
    img = cv2.imread('bed.jpg',0)
    print iclass.predict(img)
    img = cv2.imread('chair.jpg',0)
    print iclass.predict(img)
