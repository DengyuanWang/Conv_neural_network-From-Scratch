import cv2
import numpy as np
import numpy.matlib
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

def im2col(x,w_conv,b_conv):
    kh,kw,ic,oc = w_conv.shape
    h,w,_ = x.shape
    IM = np.zeros([h+kh-1,w+kw-1,ic])
    for i in range(0,x.shape[2]):
        IM[:,:,i]=np.pad(x[:,:,i], [(int(kh/2), int(kh/2)), (int(kw/2), int(kw/2))], mode='constant', constant_values=0)
    kkic = int(kh*kw*ic)
    stride = 1
    imout_h,imout_w = int(h/stride),int(w/stride)
    loc_num = [imout_h,imout_w]
    im_rows = np.zeros([loc_num[0]*loc_num[1],kkic])
    for i in range(0,loc_num[0]):
        for j in range(0,loc_num[1]):
            i_s = i*stride
            j_s = j*stride
            im_rows[i*loc_num[1]+j,:] = IM[i_s:i_s+kh,j_s:j_s+kw,:].reshape(1,-1)
    w_cols = np.zeros([kkic,oc])
    for i in range(0,oc):
        w_cols[:,i] = w_conv[:,:,:,i].reshape(-1)
    rst = np.dot(im_rows,w_cols)#rst is patch_num by output_channels
    bias = np.matlib.repmat(np.transpose(b_conv),rst.shape[0],1)
    rst = np.add(rst,bias)
    return rst,im_rows,w_cols

def conv(x, w_conv, b_conv):
    # TO DO
    #Tested, Correct!
    #x is 14 by 14 by 1
    #w_conv is 3 by 3 by 1 by 3, aka: kernel_height by kernel_width by inchannels by out channels
    #b_conv is 3 by 1, aka output channels by 1
    #assume step is 1, kernel height and width is always odd number
    #here we will use im2col
    kh,kw,ic,oc = w_conv.shape
    h,w,_ = x.shape
    stride = 1
    imout_h,imout_w = int(h/stride),int(w/stride)
    
    rst,im_rows,w_cols = im2col(x, w_conv, b_conv)
    #col2img
    y = np.zeros([imout_h,imout_w,oc])
    for i in range(0,rst.shape[1]):
        y[:,:,i] = rst[:,i].reshape(y.shape[0],y.shape[1])
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    #seems to be correct
    # y is height by witdh by channels
    #dl_dy is 1 by height by witdh by channels
    #dl_db is 1 by channels
    #dl_db_i = dl_dy[:,:,i] * ones[height, width]*dy_db
    #dl_dw is [1,kh,kw,ic,oc]
    dl_db = np.zeros([1,b_conv.shape[0]])
    #dl_drst is [height*width,oc]
    dl_drst = np.zeros([y.shape[0]*y.shape[1],y.shape[2]])
    dl_dw = np.zeros(w_conv.shape)
    _,im_rows,_ = im2col(x, w_conv, b_conv)
    #dl_dw_cols = np.zeros(w_cols.shape)
    for i in range(0,dl_dy.shape[-1]):
        dl_db[0,i] = np.sum(dl_dy[0,:,:,i])
        dl_drst[:,i] = dl_dy[0,:,:,i].reshape(-1)
        #dl_drst is height*width by out channel; im_rows is height*width by kkic,
       #dl_dw_cols[:,i] = np.dot(dl_drst[:,i],im_rows)
        dl_dw[:,:,:,i] = np.dot(dl_drst[:,i],im_rows).reshape(dl_dw.shape[0],dl_dw.shape[1],dl_dw.shape[2])
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
    #main.main_mlp()
    #main.main_cnn()



