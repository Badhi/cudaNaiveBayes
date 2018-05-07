import numpy as np
import random
from sklearn.naive_bayes import  GaussianNB
from time import time

from testNB import customNB

x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

x_rand = np.random.randint(-10, 100, size = (1000, 15))

y_rand = np.random.choice([10, 20], 1000, replace= True)

#print x
#print Y
t0 = time()
clf = GaussianNB()


customnb = customNB()

customnb.fit(x_rand,y_rand)

clf.fit(x_rand, y_rand)

#print "training time:", round(time()-t0, 3), "s"


#testData = [[1,2], [7,5], [-4, 5], [-3, 2]]
testData = np.random.randint(-4, 7, size = (100, 15))
predicted= clf.predict(testData)
customPredict = customnb.predict(testData)


print "sklearn Out"
print predicted

print "custom Out"
print customPredict

print predicted - customPredict
