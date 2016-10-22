from scipy import *
from scipy.linalg import norm, pinv
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np

X = []


clf_neutral = joblib.load('nn/net_neutral.nn')
clf_happy = joblib.load('nn/net_happy.nn')
clf_sad = joblib.load('nn/net_sad.nn')
clf_surprise = joblib.load('nn/net_surprise.nn')
clf_angry = joblib.load('nn/net_angry.nn')

def classifyEmotion(feat):
    global clf_neutral, clf_happy, clf_sad, clf_surprise, clf_angry
    X = np.asarray(feat)
    return [int(clf_neutral.predict(X.reshape(1,-1))[0]),int(clf_happy.predict(X.reshape(1,-1))[0]),int(clf_sad.predict(X.reshape(1,-1))[0]),int(clf_surprise.predict(X.reshape(1,-1))[0]),int(clf_angry.predict(X.reshape(1,-1))[0])]
