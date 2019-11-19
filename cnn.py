import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
import random
np.random.seed(2)

def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    #im_train is 196 * sample_size
    print(label_train.shape)
    print(im_train.shape)
    idicies = np.random.permutation(im_train.shape[1])
    batch_num = int(math.ceil(im_train.shape[1]/batch_size))
    mini_batch_x = np.zeros([196,batch_size,batch_num])
    mini_batch_y = np.zeros([10,batch_size,batch_num])
    count = 0;
    for i in range(0,mini_batch_x.shape[2]):
        for j in range(0,batch_size):
            if(i*batch_size+j<im_train.shape[1]):
                id = idicies[i*batch_size+j]
            else:#random fill last batch from all samples
                print("123123\n")
                id = random.randint(0, im_train.shape[1])
            mini_batch_x[:,j,i] = im_train[:,id]
            mini_batch_y[label_train[0,id],j,i] = 1
            count+=1
    print(mini_batch_x.shape)
    print(count)
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    #w is 10 by 196
    #x is 196 by 1
    #b is 10 by 196
    y = np.dot(w, x)+b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO x(196 by 1) w(10 by 196) b(10 by 1) y(10 by 1)
    # dl_dy is 1 by 10;
    # dl_dw is 196 by 10 (196 by 1 * 1 by 10)
    # dl_dx is 1 by 196 (1 by 10 * 10 by 196)
    # dl_db is 1 by 10 (1 by 10)
    
    ''' w = 10*196=[w1,w2...w196]
    dl_dw = dl/dw1_1.... dl/dw1_196
            ...
            ...
            dl/dw10_1.... dl/dw10_196,
    l = sum_i((y_i-yT_i)^2)
    dl/dw1_1 = dl/dy_1*dy_1/dw1_1 = 2(y_1-yT_1)*X1          dl/dw1_196 = dl/dy_1*dy_1/dw1_196 = 2(y_1-yT_1)*X196
    dl/dw10_1 = dl/dy_10*dy_10/dw1_1 = 2(y_10-yT_10)*X1     dl/dw10_196 = dl/dy_10*dy_10/dw1_196 = 2(y_10-yT_10)*X196
    '''
    dl_db = dl_dy
    dl_dw = np.dot(x,dl_dy)
    dl_dx = np.dot(dl_dy,w)
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    #L = sum((y_tilde_i-y_i)^2)
    #dl_dy is 1 by 10
    diff =(y_tilde-y)
    T = np.multiply(diff,diff)
    l = np.sum(T)
    dl_dy = np.transpose(2*diff)
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    '''
        y_tilde_i = e^x_i/sum_j(e^x_j)
        L = sum_i(y_i*log(y_tilde_i))
        dL_dx_i =sum(dL_dy_tilde_j * dy_tilde_jdx_i)
        dL_dy_tilde_j = y_j/y_tilde_j
        dy_tilde_jdx_i =
                        for i==j
                        y = a/(a+b); a = e^x_i
                        dy_dx_i = dy_da*da_dx_i = ((a+b)-a(1))/(a+b)^2 * da_dx_i
                                                = b/(a+b)^2 * e^x_i
                                                = (sum()-e^x_i)/(sum()^2) * e^x_i
                        for i!=j
                        y = c/(a+b); a=e^x_i
                        dy_dx_i = dy_da*da_dx_i = -c/(a+b)^2 * da_dx_i = -e^x_j/sum()^2 * e^x_i
        []*[] = for i==j
                    y_j*e^x_i* (sum-e^x_i)/(e^x_i*sum)
                    = y_j - y_j*e^x_i/sum
                for i!=j
                    -y_j*e^x_i/sum
        sum([]*[]) = sum([0 0 0 0 y_i 0 0 0] - [..y_j-1 y_j y_j+1...]*e^x_i/sum)
    '''
    cc = np.array(x, dtype=np.float128)
    ex = np.exp(cc)
    sum_ex = np.sum(ex)
    y_tilde = ex/sum_ex
    l = -np.sum(np.multiply(y,np.log(y_tilde)))
    tmp = np.zeros([x.shape[0],x.shape[0]])
    np.fill_diagonal(tmp,y)
    dl_dy =tmp - np.dot(y_tilde,np.transpose(y))

    dl_dy = -np.sum(dl_dy,axis=1)
    dl_dy = dl_dy.reshape(-1,1)
    #dl_dy is 1 by n
    dl_dy = np.transpose(dl_dy)
    return l, dl_dy

def relu(x):
    # TO DO
    # if x>0: y = x
    # if x<=0 y = 0.01 * x
    x[x<=0] *= 0.01
    y = x
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    #dl_dy(1 by n) x(n by 1)
    # if x>0: y = x
    # if x<=0 y = 0.01 * x
    dy_dx = np.transpose(np.where(x > 0, 1, 0.01))
    dl_dx = np.multiply(dl_dy,dy_dx)
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    gama = 0.05 # learning rate
    lambda_ = 0.9 # decay rate

    mu, sigma = 0, 1 # mean and standard deviation
    w = np.random.normal(mu, sigma, [10,196])
    wmax = np.max(np.max(np.absolute(w), axis=0), axis=0)
    w = w/wmax
    b = np.random.normal(mu, sigma, [10,1])
    bmax = np.max(np.max(np.absolute(b), axis=0), axis=0)
    b = b/bmax
    batch_id = 0
    batch_size = mini_batch_x.shape[1]
    L_log = []
    for iter in range(0,20000):
        if (iter+1)%1000==0:
            gama *= lambda_
            print(gama)
        dL_dw = 0
        dL_db = 0
        L = 0
        for i in range(0,batch_size):
            x = mini_batch_x[:,i,batch_id].reshape(-1,1)
            y = mini_batch_y[:,i,batch_id].reshape(-1,1)
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_euclidean(y_tilde,y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw += dl_dw
            dL_db += dl_db
            L = L+l
        batch_id = (batch_id+random.randint(0, mini_batch_x.shape[2]) )%mini_batch_x.shape[2]
        w -= gama*np.transpose(dL_dw)/batch_size
        b -= gama*np.transpose(dL_db)/batch_size
        #L_log.append(L/batch_size)
    print("Train Over for SLP_Linear")
    '''
    plt.figure(1)
    plt.plot(np.array(L_log))
    plt.draw()
    '''
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    gama = 0.05 # learning rate
    lambda_ = 0.9 # decay rate
    
    mu, sigma = 0, 1 # mean and standard deviation
    w = np.random.normal(mu, sigma, [10,196])
    wmax = np.max(np.max(np.absolute(w), axis=0), axis=0)
    w = w/wmax
    b = np.random.normal(mu, sigma, [10,1])
    bmax = np.max(np.max(np.absolute(b), axis=0), axis=0)
    b = b/bmax
    batch_id = 0
    batch_size = mini_batch_x.shape[1]
    L_log = []
    for iter in range(0,20000):
        if (iter+1)%1000==0:
            gama *= lambda_
            print(gama)
        dL_dw = 0
        dL_db = 0
        L = 0
        for i in range(0,batch_size):
            x = mini_batch_x[:,i,batch_id].reshape(-1,1)
            y = mini_batch_y[:,i,batch_id].reshape(-1,1)
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde,y)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            dL_dw += dl_dw
            dL_db += dl_db
            L = L+l
        batch_id = (batch_id+random.randint(0, mini_batch_x.shape[2]) )%mini_batch_x.shape[2]
        w -= gama*np.transpose(dL_dw)/batch_size
        b -= gama*np.transpose(dL_db)/batch_size
        #L_log.append(L/batch_size)
    print("Train Over for SLP")
    '''
    plt.figure(2)
    plt.plot(np.array(L_log))
    plt.draw()
    '''
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    gama = 0.05 # learning rate
    lambda_ = 0.9 # decay rate
    
    mu, sigma = 0, 1 # mean and standard deviation
    w1 = np.random.normal(mu, sigma, [30,196])
    w1 = w1/np.max(np.max(np.absolute(w1), axis=0), axis=0)
    b1 = np.random.normal(mu, sigma, [30,1])
    b1 = b1/np.max(np.max(np.absolute(b1), axis=0), axis=0)
    w2 = np.random.normal(mu, sigma, [10,30])
    w2 = w2/np.max(np.max(np.absolute(w2), axis=0), axis=0)
    b2 = np.random.normal(mu, sigma, [10,1])
    b2 = b2/np.max(np.max(np.absolute(b2), axis=0), axis=0)
    
    batch_id = 0
    batch_size = mini_batch_x.shape[1]
    L_log = []
    for iter in range(0,20000):
        if (iter+1)%1000==0:
            gama *= lambda_
            print(gama)
        dL_dw1 = 0
        dL_db1 = 0
        dL_dw2 = 0
        dL_db2 = 0
        L = 0
        for i in range(0,batch_size):
            x = mini_batch_x[:,i,batch_id].reshape(-1,1)
            y = mini_batch_y[:,i,batch_id].reshape(-1,1)
            y0_tilde = fc(x, w1, b1)
            y0_relu = relu(y0_tilde)
            y1_tilde = fc(y0_relu, w2, b2)
            y1_relu = relu(y1_tilde)
            l, dl_dy = loss_cross_entropy_softmax(y1_relu,y)
            #dl_dx = relu_backward(dl_dy, x, y)
            #dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y)
            #layer2
            dl_dy1_tilde = relu_backward(dl_dy, y1_tilde, y1_relu)
            dl_dy0_relu, dl_dw2, dl_db2 = fc_backward(dl_dy1_tilde, y0_relu, w2, b2, y1_tilde)
            #layer1
            dl_dy0_tilde= relu_backward(dl_dy0_relu, y0_tilde, y0_relu)
            dl_dx, dl_dw1, dl_db1 = fc_backward(dl_dy0_tilde, x, w1, b1, y0_tilde)
           
            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2
            L = L+l
        batch_id = (batch_id+random.randint(0, mini_batch_x.shape[2]) )%mini_batch_x.shape[2]
        w1 -= gama*np.transpose(dL_dw1)/batch_size
        b1 -= gama*np.transpose(dL_db1)/batch_size
        w2 -= gama*np.transpose(dL_dw2)/batch_size
        b2 -= gama*np.transpose(dL_db2)/batch_size
        L_log.append(L/batch_size)
    print("Train Over for SLP")
    '''
    plt.figure(1)
    plt.plot(np.array(L_log))
    plt.draw()
    plt.pause(5)
    plt.clf
    '''
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    #main.main_slp_linear()
    #main.main_slp()
    main.main_mlp()
    main.main_cnn()



