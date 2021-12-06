"""
Created on Thu Dec 12 16:49:04 2019

@author: yunus
"""
import numpy as np
def location(index,num_det,l,teta):
    l=l-1
    a=l/(np.pi/2)
    b=l/num_det
    y=0
    x_1=index*b+teta*a
    x=l-x_1 
    if(x_1>l):
        y=(x_1-l)
        x=0
    if(y>l):
        x=np.abs(l-y)
        y=l
        
    #print(x,y)
    return x ,y
            
def p_values(matrix,j,length,num_of_det,t):
    a=0
    for i in range(length):
        x_1,y_1=location(j,num_of_det,length,t)
        #print(y_1)
        x_e=np.round(x_1+i*np.sin(t))
        y_e=np.round(y_1+i*np.cos(t))
        #print(x_e)

        if x_e==length :
            break
        if y_e==length:
            break
        x_pix=int(np.round(x_1+i*np.sin(t))%length)
        y_pix=int(np.round(y_1+i*np.cos(t))%length)
        a+=matrix[y_pix,x_pix] 
        #print(matrix[y_pix,x_pix])
        #print(x_pix,y_pix)
    return a 

#from PIL import Image
from skimage.draw import circle
from skimage.draw import rectangle
img = np.zeros((70, 70), dtype=np.uint8)
start = (20,20 )
extent = (10, 10)
rr, cc = rectangle(start, extent=extent, shape=img.shape)
img[rr, cc] = 1
img[60,60]=1
#im = Image.open()
matrix=img
teta=180
intensity=1
l=11
c_1=p_values(matrix,43,50,50,3*np.pi/4)
x_2,y_2= location(0,50,10,3*np.pi/4)
#import skimage.draw as draw
#matrix = np.array(im)
p = []

for t in np.arange(0,np.pi,0.5*(np.pi/180)):
    b=[]
    for index_of_det in range(50):
        a=p_values(matrix,index_of_det,50,50,t)
        b.append(a)
        
    p.append(b)
p=np.array(p)
p=np.transpose(p)
import matplotlib.pyplot as plt

def show_p(matrix,l,num_det,t):
    b=[]
    for index_of_det in range(num_det):
            a=p_values(matrix,index_of_det,50,50,t)
            b.append(a)
        
    t_1=np.array(b)
    plt.plot(t_1) # plotting by columns
    plt.show()
    return t_1
v=[0,np.pi/4,np.pi/2,3*np.pi/4,np.pi]

for t in v:
    show_p(matrix,50,50,t)
    
d=p[91]
f=p.shape
#for i in p:
#    print(i)
def back_project(size,p,length):
    a,l=p.shape
    matrix=np.zeros(size)
    for t,p_x in enumerate (p,0):
        t=t*np.pi/180
        #print(t)
    
        for i in range(l-1):
            s=p_x[i]
            for j in range(length):
                x_1,y_1=location(i,l,length,t)
        #print(y_1)
                x_e=np.round(x_1+j*np.sin(t))
                y_e=np.round(y_1+j*np.cos(t))
        #print(x_e)

           
                x_pix=int(np.round(x_1+j*np.sin(t))%length)
                y_pix=int(np.round(y_1+j*np.cos(t))%length)
                #print(x_pix,y_pix)

                matrix[y_pix,x_pix]=+s
            
    
    return matrix

mat_1=back_project((50,50),p,50)
"Imports"
