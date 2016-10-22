from scipy import *
from scipy.linalg import norm, pinv
from sklearn.svm import SVC
from sklearn.externals import joblib
import sklearn.gaussian_process as gpml
import numpy as np

X = []
Y_happy = []
Y_sad = []
Y_angry = []
Y_neutral = []
Y_surprise = []

if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------



    #classificadores
    clf_happy = SVC(kernel='linear',probability=True, tol=1e-3)
    clf_sad = SVC(kernel='linear',probability=True, tol=1e-3)
    clf_angry = SVC(kernel='linear',probability=True, tol=1e-3)
    clf_neutral = SVC(kernel='linear',probability=True, tol=1e-3)
    clf_surprise = SVC(kernel='linear',probability=True, tol=1e-3)


    #file manager
    file = open("features.xls", "r")
    table = file.readlines()


    del table[0]
    for line in table:
    	aux = line.split()

    	valuesFloat = []
    	for valuesStr in aux[0:len(aux)-1]:
    		valuesFloat.append(float(valuesStr))
    	X.append(valuesFloat)
        if(int(aux[len(aux)-1]) == 0):   #Neutro
            Y_neutral.append(1)
            Y_happy.append(-1)
            Y_sad.append(-1)
            Y_surprise.append(-1)
            Y_angry.append(-1)

        if(int(aux[len(aux)-1]) == 1):   #Feliz
            Y_neutral.append(-1)
            Y_happy.append(1)
            Y_sad.append(-1)
            Y_surprise.append(-1)
            Y_angry.append(-1)


        if(int(aux[len(aux)-1]) == 2):   #Triste
            Y_neutral.append(-1)
            Y_happy.append(-1)
            Y_sad.append(1)
            Y_surprise.append(-1)
            Y_angry.append(-1)


        if(int(aux[len(aux)-1]) == 3):   #surpresa
            Y_neutral.append(-1)
            Y_happy.append(-1)
            Y_sad.append(-1)
            Y_surprise.append(1)
            Y_angry.append(-1)

        if(int(aux[len(aux)-1]) == 4):   #raiva
            Y_neutral.append(-1)
            Y_happy.append(-1)
            Y_sad.append(-1)
            Y_surprise.append(-1)
            Y_angry.append(1)

    file.close()

    X = np.asarray(X)
    Y_neutral = np.asarray(Y_neutral)
    Y_happy = np.asarray(Y_happy)
    Y_sad = np.asarray(Y_sad)
    Y_surprise = np.asarray(Y_surprise)
    Y_angry = np.asarray(Y_angry)


    #treinamento
    clf_neutral.fit(X, Y_neutral)
    clf_happy.fit(X, Y_happy)
    clf_sad.fit(X, Y_sad)
    clf_surprise.fit(X, Y_surprise)
    clf_angry.fit(X, Y_angry)


    #salvando em arquivos
    joblib.dump(clf_neutral, 'nn/net_neutral.nn')
    joblib.dump(clf_happy, 'nn/net_happy.nn')
    joblib.dump(clf_sad, 'nn/net_sad.nn')
    joblib.dump(clf_surprise, 'nn/net_surprise.nn')
    joblib.dump(clf_angry, 'nn/net_angry.nn')

