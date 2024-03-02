import torch
import sklearn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import os
dd=os.listdir('TIN')
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
for i in range(len(dd)):
    d2 = os.listdir ('TIN/%s/images/'%(dd[i]))
    for j in range(len(d2)-2):
        str1='TIN/%s/images/%s'%(dd[i], d2[j])
        f1.write("%s %d\n" % (str1, i))
    str1='TIN/%s/images/%s'%(dd[i], d2[-1])
    f2.write("%s %d\n" % (str1, i))

f1.close()
f2.close()



import numpy as np
from numpy import linalg as LA
import cv2

def load_img(f):
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        
        im1=cv2.imread(fn)
        im1=cv2.resize(im1, (256,256))
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        
        #===============================
        #影像處理的技巧可以放這邊，來增強影像的品質
        im1 = fastNlMeansDenoising()
        
        #===============================
        
        vec = np.reshape(im1, [-1])
        imgs.append(vec) 
        lab.append(int(label))
        
    imgs= np.asarray(imgs, np.float32)
    lab= np.asarray(lab, np.int32)
    return imgs, lab 


x, y = load_img('train.txt')
tx, ty = load_img('test.txt')


from sklearn.neighbors import KNeighborsClassifier

# Define and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(x, y)

# Evaluate KNN classifier
accuracy = knn_classifier.score(tx, ty)
print("Accuracy:", accuracy)


from sklearn.svm import SVC
# Define and train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(x, y)

# Evaluate SVM classifier
accuracy = svm_classifier.score(tx, ty)
print("Accuracy:", accuracy)
