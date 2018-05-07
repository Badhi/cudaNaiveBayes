import numpy as np
from cudaNB import CudaNB
import math

class customNB :
    bins = 13
    classes = []
    class_counts = 0
    feature_vals = []
    feature_val_counts = 0

    classmap = {}
    featuremap = {}

    featuresGivenResults = []

    def getFeaturesElements(self, x) :
        normalizedFeatureElements = np.linspace(np.amin(x), np.amax(x), self.bins)
#        normalizedFeatureCounts = np.zeros(len(normalizedFeatureElements))

#        flatArray = x.flatten()
#        for arVal in flatArray :
#           normalizedFeatureCounts[np.abs(normalizedFeatureElements - arVal).argmin()] += 1

        cudanb = CudaNB()
        normalizedFeatureCounts = cudanb.histogram(x, normalizedFeatureElements, isGPUmode = True)

        return normalizedFeatureCounts, normalizedFeatureElements

    def getFeatureElementsFromHist(self, x):
        return np.histogram(x, bins = self.bins)


    def getBinaryArray(self, value, max):
        return map(int ,list("{0:b}".format(2**value).zfill(max)))

    def decodeKey(self, array):
        #print("array len : {}".format(len(array)))

        key = 0
        latch = False
        for i in array :
            if latch == True :
               key += 1
            if i == 1 :
                latch = True

        return key
#        val = 0
#        for i in array :
#            val = val * 2 + i

        return int(math.log(val, 2))

    def getNearestFeature(self, value):
        return self.feature_vals[np.abs(self.feature_vals - value).argmin()]

    def getFeatureIndex(self, feature_val):
        #print(self.featuremap[self.getNearestFeature(feature_val)])
        return self.decodeKey(self.featuremap[self.getNearestFeature(feature_val)])
    def getClassIndex(self, class_val):
        return self.decodeKey(self.classmap[class_val])

    def fit(self, x, y) :
        x_arr = np.array(x)
        y_arr = np.array(y)


        self.classes, self.class_counts = np.unique(y_arr, return_counts = True)
#        self.feature_vals, self.feature_val_counts = np.unique(x_arr, return_counts = True)

        self.feature_val_counts, self.feature_vals  = self.getFeaturesElements(x_arr)

        print("Features")
        print(self.feature_vals)
        print(self.feature_val_counts)

        index = 0
        for i in self.classes :
            self.classmap[i] = self.getBinaryArray(index, len(self.classes))
            index += 1

        index = 0
        for i in self.feature_vals :
            self.featuremap[i] = self.getBinaryArray(index, self.bins + 1)
            index += 1


        label_vals = np.array([self.classmap[i] for i in y_arr])
        feature_vals = np.zeros((len(y_arr), len(self.feature_val_counts)), dtype = int)

        k = 0
        for i in x_arr :
            val = self.getBinaryArray(0, len(self.feature_val_counts))
            val[-1] = 0
            for j in i :
                val = [a + b for a,b in zip(val, self.featuremap[self.getNearestFeature(j)])]

            feature_vals[k] = val
            k+=1

#        print(x_arr)
#        print(y_arr)

#        print(self.featuremap)
#        print(self.classmap)

#        print(np.transpose(feature_vals))
#        print(label_vals)
        cudanb = CudaNB()
        self.featuresGivenResults =  np.flipud(cudanb.matMultiply(np.transpose(feature_vals), label_vals, isGPU=True))

        print(self.featuresGivenResults)

    def predict(self, test_data):

        classToIndexMap = {}
        class_prob = np.ones((len(self.classes), 1))
        labels_result = []
        for instance in test_data :
            for class_val in self.classes:
                class_index = self.getClassIndex(class_val)

                given = 1

                for feature in instance :
                    feature_index = self.getFeatureIndex(feature)
                    given *= self.featuresGivenResults[feature_index][class_index]

                class_prob[class_index] = given * self.class_counts[class_index]
                classToIndexMap[class_index] = class_val

            #print(class_prob)
            #print(classToIndexMap[class_prob.argmax()])
            labels_result.append(classToIndexMap[class_prob.argmax()])

        return labels_result
#        print(self.classes)
#        print(self.class_counts)
#        print(self.feature_vals)
#        print(self.feature_val_counts)
#        print(self.classmap.values())



