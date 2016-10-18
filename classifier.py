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

def classifyEmotion(feat):
    X = np.asarray(feat[0:-1])
    # classe = switch(int(feat[-1]))
    clf = joblib.load('net.nn')
   # clf.fit(X,Y)
    acertou = 0
    print "classificou como ", str(clf.predict(X.reshape(1,-1))[0])

    # if classe == predict:
    #     print "acertou ", classe
    #     acertou += 1
    # print "****************************************************"
    # print clf.score(X,classe)



if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------



    #file manager
    file = open("predict.xls", "r");
    table = file.readlines()

    del table[0]
    for line in table:
    	aux = line.split()
    	valuesFloat = []
    	for valuesStr in aux:
    		valuesFloat.append(float(valuesStr))
    	X.append(valuesFloat)
    	# X.append(switch(int(aux[len(aux)-1])))
    file.close()
    # Y = np.asarray(Y)
    for i in X:
        classifyEmotion(i)
# for pos, entrada in enumerate(X):
#     print "classificou ", Y[pos], "como ", clf.predict(X[pos].reshape(1,-1))
#     if Y[pos] == clf.predict(entrada.reshape(1,-1)):
#        print "acertou ", pos, Y[pos]
#        acertou += 1
#     print "****************************************************"
# print "acertou ",  (100*acertou)/len(X), "%", " das imagens"

# print clf.score(X,Y)
