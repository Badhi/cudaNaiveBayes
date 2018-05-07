#! /x01/bhashithaa/anaconda3/bin/python

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpu_array
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time

class CudaNB:
    def histoGPU(self, data, bins):

        histofun_mod = SourceModule("""
                        #include <cstdio>
                        __global__ void histogram(float * data, float * bins, int * freqCount, int binCount, int dataCount)
                        {
                            int tid = threadIdx.x + blockDim.x * blockIdx.x;
                              
                            for(int i = tid; i < binCount; i+=blockDim.x) 
                            {
                                freqCount[i] = 0;
                            }

                            if(tid < dataCount)
                            {
                                float threadData = data[tid];
                                int nearestIndex = 0;

                                for(int i = 0; i < binCount; i++)
                                {
                                   if(abs(threadData - bins[i]) < abs(threadData - bins[nearestIndex]))
                                        nearestIndex = i; 
                                }

                                //printf("%d, %f, %d\\n", tid, threadData, nearestIndex);

                                atomicAdd(&freqCount[nearestIndex], 1);
                            }
                        }
                        """)


        histogramFun = histofun_mod.get_function("histogram")

        blockSize = 64 

        if(data.size % blockSize == 0):
            gridSize = int(data.size/blockSize)
        else :
            gridSize = int(data.size/blockSize) + 1

        frequencyCount = np.zeros(bins.size).astype(np.int32)
#        print(frequencyCount)

#        print("Blocks")
#        print(blockSize)
#        print("grids")
#        print(gridSize)

        #print(flatData)

        histogramFun(cuda.In(data), cuda.In(bins.astype(np.float32)), cuda.Out(frequencyCount), np.int32(bins.size), np.int32(data.size), block=(blockSize, 1,1), grid=(gridSize, 1, 1))

        return frequencyCount

    def histoCPU(self, data, bins):
        hisAr = np.zeros(len(bins))

        for dataVal in data:
            hisAr[np.abs(bins - dataVal).argmin()] += 1

        return hisAr

    def histogram(self, data, bins, isGPUmode=False):
        if(len(bins.shape) != 1):
            print("Bins should be 1D")
            return 

        flatData = data.flatten().astype(np.float32)

        if(isGPUmode == True) :
            return self.histoGPU(flatData,bins)
        else:
            return self.histoCPU(flatData,bins)


    def matMultiply(self, a, b, isGPU = False):
        if(len(a.shape) != 2 or len(b.shape) != 2 ) :
            print("Input data are not matrices")
            return

        if(a.shape[1] != b.shape[0]):
            print("Dimensions missmatch")
            return

        if(isGPU == True) :
            return self.matMultiplyGPU(a, b)
        else :
            return self.matMultiplyCPU(a, b)


    def matMultiplyCPU(self, a, b):
        return np.matmul(a, b)

    def matMultiplyGPU(self, a, b):

        rows = a.shape[0]
        cols = b.shape[1]

        commonDim = a.shape[1]


        multiplyfun = SourceModule("""
                        #include <cstdio>
                        __global__ void matMultiply(int * a, int * b, int * c, int commonVal, int X, int Y)
                        {
                            int xPos = threadIdx.x + blockIdx.x * blockDim.x;
                            int yPos = threadIdx.y + blockIdx.y * blockDim.y;

                            if(xPos < X && yPos < Y)
                            {
                                int sum = 0;
                                for(int i = 0; i < commonVal; i++)
                                {
                                    //printf("%d, %d, %d, %d, %d\\n", i, xPos, yPos, a[xPos * commonVal + i], b[i * commonVal + yPos]);
                                    sum += a[xPos * commonVal + i] * b[i * Y + yPos];
                                }

                                c[xPos * Y + yPos] = sum; 
                            }
                        }
                        """)

        matMulti = multiplyfun.get_function("matMultiply")

        c = np.zeros((rows, cols)).astype(np.int32)

        blockXSize = 32
        blockYSize = 32
        blockSize = (blockXSize, blockYSize, 1)

        if (rows % blockXSize == 0) :
            gridXSize = int(rows/blockXSize)
        else :
            gridXSize = int(rows/blockXSize + 1)

        if (cols % blockYSize == 0) :
            gridYSize = int(cols/blockYSize)
        else :
            gridYSize = int(cols/blockYSize)  + 1

        gridSize = (gridXSize, gridYSize, 1)

        matMulti(cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(commonDim), np.int32(rows), np.int32(cols), block=blockSize, grid=gridSize)

        return c

def matMutliTest(cudanb) :
    a = np.random.randint(50, 100, size=(32100,100)).astype(np.int32)
    b = np.random.randint(25, 300, size=(100,31900)).astype(np.int32)

    #    print(a)
    #    print(b)
    np.savetxt("a.csv", a, delimiter=',')
    np.savetxt("b.csv", b, delimiter=',')

    start = time.time()
    c = cudanb.matMultiply(a,b, isGPU=True)
    end = time.time()

    print("GPU Time : {}".format((end - start)* 1000))

    start = time.time()
    c_cpu = cudanb.matMultiply(a,b)
    end = time.time()

    print("CPU Time : {}".format((end - start) * 1000))

    print("Error : {}".format(sum(sum(c - c_cpu))))

    np.savetxt("c.csv", c, delimiter=',')
    #print(c)

def histoTest(cudanb):
#    a = np.array([5, 4, 10, 15, 25, 33, 35, 34, 39, 45, 55, 100])
    a = np.random.randint(-11, 1000, size=(10000000))
    bins = np.linspace(a.min(), a.max(), 10)

    start = time.time()
    #binCountsHost = cudanb.histogram(a, bins)
    end = time.time()

    print("CPU Time : {}".format((end - start) * 1000))

    start = time.time()
    binCounts = cudanb.histogram(a, bins, isGPUmode=True)
    end = time.time()

    print("GPU Time : {}".format((end - start) * 1000))

    #print(a)
    #print(bins)
    #print(binCounts)
    #print(binCountsHost)

    #print("Error : {}".format(sum(binCounts - binCountsHost)))

if __name__ == "__main__" :
    cudanb = CudaNB()

    np.set_printoptions(threshold=np.nan)

#    matMutliTest(cudanb)
    histoTest(cudanb)

    cuda.stop_profiler()
