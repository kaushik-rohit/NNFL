import numpy as np
import cv2

def gaussian_filter(sd):
    fil = np.zeros((5,5))
    
    sum = 0
    
    for x in range(-2,3):
        for y in range(-2,3):
            r = x**2 + y**2
            d = 2*(sd**2)
            fil[x+2][y+2] = np.exp(-1*r/d)*(1/(sd**2)*np.pi)
            sum += fil[x+2][y+2]
    
    for i in range(5):
        for j in range(5):
            fil[i][j] /= sum
            
    return fil
    
        
def gaussian_blur(img):
     imgx, imgy = img.shape
     fil = gaussian_filter(1.4)
     filx, fily = fil.shape
     print fil.shape
     stride = 1
     
     for x in range(imgx):
        for y in range(imgy):
            sume = 0
            print (x,y)
            for i in range(x - filx/2, x + filx/2):
                for j in range(y - fily/2, y + fily/2):
                    if(i < 0 or j < 0 or i >= imgx or j >= imgy):
                        continue
                    sume += img[i][j]*fil[i - x - filx/2]
            #print img[x][y]
            img.put((x,y), sume)
    
     return img
     
def edge_detect(img):
    pass
    
    
if __name__ == "__main__":
    img = cv2.imread('test.jpg',0)
    #print img
    img2 = gaussian_blur(img)
    
    cv2.imwrite('fail.jpg',img2)
    
