import numpy as np

def contranorma(X):
    
    im_maxR=max(X[0::3])
    im_minR=min(X[0::3])
    im_maxG=max(X[1::3])
    im_minG=min(X[1::3])
    im_maxB=max(X[2::3])
    im_minB=min(X[2::3])
    Xsize = np.shape(X)[0]
    
    minR = np.zeros((1,Xsize/3))
    minR[0::3]=1
    minG = np.zeros((1,Xsize/3))
    minG[1::3]=1
    minB = np.zeros((1,Xsize/3))
    minB[2::3]=1
    
    
    X[0::3] = (X[0::3] - minR)*1.0/(im_maxR-im_minR)
    X[1::3] = (X[1::3] - minG)*1.0/(im_maxG-im_minG)
    X[2::3] = (X[2::3] - minB)*1.0/(im_maxB-im_minB)
    
    return X

def crop3(A):
    length = 32
    newlength = 28
    channel=3
    x = np.random.randint(0,length-newlength+1)
    y = np.random.randint(0,length-newlength+1)
    B = np.zeros((1,newlength*newlength*channel))
    for i in range (0,newlength):
        #print str((i*newlength+newlength)*channel-i*newlength*channel)
        #print str((y*length+i*length+x+newlength)*channel-(y*length+i*length+x)*channel)
        #print i        
        B[0,i*newlength*channel:(i*newlength+newlength)*channel]=A[(y*length+i*length+x)*channel:(y*length+i*length+x+newlength)*channel]
    #print str(B.shape)
    return B.reshape(newlength*newlength*channel)


