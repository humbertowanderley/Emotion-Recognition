from scipy import *
from scipy.linalg import norm, pinv
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np

X = []
Y = []

def switch(argument):
    switcher = {
        0: "neutro",
        1: "feliz",
        2: "triste",
        3: "surpresa",
        4: "raiva"
    }
    return switcher.get(argument, "nothing")

clf = joblib.load('nn/net.nn')

def classifyEmotion(feat):
    global clf
    X = np.asarray(feat[0:-1])
    # classe = switch(int(feat[-1]))
   # clf.fit(X,Y)
    return str(clf.predict(X.reshape(1,-1))[0])
    # print clf.score(X,classe)
