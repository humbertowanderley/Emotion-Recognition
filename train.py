from scipy import *
from scipy.linalg import norm, pinv
from sklearn.svm import SVC
from sklearn.externals import joblib
import sklearn.gaussian_process as gpml
import numpy as np

X = []
Y = []
Y_happy = []
Y_sad = []
Y_angry = []
Y_neutral = []
Y_surprise = []

if __name__ == '__main__':



    #classificadores
    clf_happy = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.2,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf_sad = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf_angry = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf_neutral = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf_surprise = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    clf = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.1, verbose=False)

    #file manager
    file = open("features.xls", "r")
    table = file.readlines()


    del table[0]
    cont = 0;
    for line in table:
     #   print line, cont
      #  cont = cont + 1
    	aux = line.split()

    	valuesFloat = []
    	for valuesStr in aux[0:len(aux)-1]:
    		valuesFloat.append(float(valuesStr))
    	X.append(valuesFloat)
        Y.append(aux[len(aux)-1])
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
    Y = np.asarray(Y)

    #treinamento
    clf.fit(X,Y)
    clf_neutral.fit(X, Y_neutral)
    clf_happy.fit(X, Y_happy)
    clf_sad.fit(X, Y_sad)
    clf_surprise.fit(X, Y_surprise)
    clf_angry.fit(X, Y_angry)
    
    # print "tamaho: ", len(X[174])

    a = [0,0.578234,0.184795, 1, 0.919414, 0.453753, 0.427511, 0.726822]
    a = np.asarray(a) 
   
    print "Neutral: ", clf_neutral.score(X,Y_neutral)
    print "Happy: ", clf_happy.score(X,Y_happy)
    print "Sad: ", clf_sad.score(X,Y_sad)
    print "Angry: ", clf_angry.score(X,Y_angry)
    print "Surprise: ", clf_surprise.score(X,Y_surprise)

    print"ALL: ", clf.score(X,Y) 
    #salvando em arquivos
    joblib.dump(clf_neutral, 'nn/net_neutral.nn')
    joblib.dump(clf_happy, 'nn/net_happy.nn')
    joblib.dump(clf_sad, 'nn/net_sad.nn')
    joblib.dump(clf_surprise, 'nn/net_surprise.nn')
    joblib.dump(clf_angry, 'nn/net_angry.nn')
    joblib.dump(clf, 'nn/net.nn')

