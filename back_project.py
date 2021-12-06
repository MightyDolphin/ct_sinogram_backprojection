# -*- coding: utf-8 -*-
"""
Created on Sun Jan 5 17:36:14 2020

@author: yunus
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from scipy.fftpack import fft, fftshift, ifft
def p_matrix(X,Y,theta,n,imageLen,p):
    rot_1= X*np.sin(theta[n])-Y*np.cos(theta[n]) 
        #print(Xrot)
    rotCor = np.round(rot_1+imageLen/2)
        #print(XrotCor)
    rotCor = rotCor.astype('int')
    pMatrix = np.zeros((imageLen, imageLen))
    x, y = np.where((rotCor >= 0) & (rotCor <= (imageLen-1))) 
    #rint(len(rotCor[x, y]))
    s = p[:,n] 
    pMatrix[x, y] = s[rotCor[x, y]]
    return pMatrix

def backproject(p, theta):
    
    imageLen = p.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))
    
    x = np.arange(imageLen)-imageLen/2
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    plt.ion()
    fig2, ax = plt.subplots()
    im = plt.imshow(reconMatrix, cmap='gray')

    theta = theta*np.pi/180
    numAngles = len(theta)

    for n in range(numAngles):
        projMatrix =  p_matrix(X,Y,theta,n,imageLen,p)  
        reconMatrix += projMatrix
        im.set_data(Image.fromarray((reconMatrix-np.min(reconMatrix))/np.ptp(reconMatrix)*255))

         
        backprojArray = np.flipud(reconMatrix)
    return backprojArray
def fft_translate(projs):
    return np.fft.rfft(projs, axis=1)



def ramp_filter(ffts):
    ramp = np.floor(np.arange(0.5, ffts.shape[0]//2 + 0.1, 0.5))
    return ffts * ramp

def inverse_fft_translate(operator):
    return np.fft.irfft(operator, axis=1)

d = 0.5
theta = np.arange(0,179,d)

def arange2(start, stop=None, step=1):
    
    if stop == None:
        a = np.arange(start)
    else: 
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:   
            a = np.delete(a, -1)
    return a
def projFilter(sino):
    a = 0.1
    projLen, numAngles = sino.shape
    step = 2*np.pi/projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1]+step]]) 
    rn1 = abs(2/a*np.sin(a*w/2))
    rn2 = np.sin(a*w/2)/(a*w/2)
    r = rn1*(rn2)**2
    filt = fftshift(r)   
    filtSino = np.zeros((projLen, numAngles))
    for i in range(numAngles):
        projfft = fft(sino[:,i])
        filtProj = projfft*filt
        filtSino[:,i] = np.real(ifft(filtProj))
    return  filtSino
if __name__ == '__main__':
    p=np.array(p)
    dTheta = 1
    theta = np.arange(0,179,dTheta)
    p_1 = projFilter(p)  
    recon_or=backproject(p, theta)
    recon = backproject(p_1, theta)
    recon2 = np.round((recon-np.min(recon))/np.ptp(recon)*255)
    reconImg = Image.fromarray(recon2.astype('uint8'))
    fig3, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.imshow(recon_or, cmap='gray')
    ax2.imshow(matrix, cmap='gray')
    plt.show()