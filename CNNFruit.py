# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:17:16 2020

@author: han
"""


import tensorflow as tf
import numpy as np
import os
import skimage
import keras
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_small_data(dir_path,m,flag):
    images_m=[] ##新建一个空列表用于存放图片数集
    labels_m=[] ##新建一个空列表用于存放标签数集
    images_show=[]
    lab_show=[]
    lab=os.listdir(dir_path)
    n=0
    for l in lab:
        if(n>=m):
            break
        img=os.listdir(dir_path+l) ##img为对应路径下的文件夹
        if flag==True:
            images_show.append(skimage.data.imread(dir_path+l+'/'+img[0]))
            lab_show.append(l)
        for i in img:
            img_path=dir_path+l+'/'+i ##是的话获取图片路径
            labels_m.append(int(n)) ##将图片的上层文件夹转换为int类型存于labels中
            images_m.append(skimage.data.imread(img_path)) ##读取对应路径图像存放于images_m中
        n+=1
    if flag==True:
        return images_m,labels_m,images_show,lab_show ## m类标签以及数据
    else:
        return images_m,labels_m

def display_no5_img(no5_imgs,labels_no5):
    
    plt.figure(figsize=(15,15)) ##显示的尺寸为15*15
    for i in range(len(no5_imgs)):
        plt.subplot(5,1,(i+1)) ##显示为11行，每行7个
        plt.title("%s" %(labels_no5[i])) ##显示标题
        plt.imshow(no5_imgs[i])  ##显示图片
        plt.axis('off') ##不显示坐标轴
    plt.show()
    
##预处理数据函数（数组化，乱序）
def prepare_data(images,labels,n_classes):
    ##images64=cut_image(images,64,64) ##裁剪图片大小为64*64
    train_x=np.array(images)
    train_y=np.array(labels)
    ##images_gray=color.rgb2gray(images_a) ##转灰度
    indx=np.arange(0,train_y.shape[0])   #生成列表0到图片大小的数量
    indx=shuffle(indx)                 #shuffle() 方法将序列的所有元素随机排序。
    train_x=train_x[indx]
    train_y=train_y[indx]             #生成乱序训练集
    train_y=keras.utils.to_categorical(train_y,n_classes) ##one-hot独热编码
#    print(train_y)
#    print(np.shape(train_y))
    return train_x,train_y

def class_label(list_num, test_x):
    label = []
    lab=os.listdir("./Test")
    for i in np.arange(5):
        label.append(lab[list_num[i]])
    display_no5_img(test_x[0:5], label)
## 定义卷积层的生成函数
def conv2d(x,W,b,stride=1):
    x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
 
## 定义池化层的生成函数
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")
 
## 定义卷积神经网络生成函数
def conv_net(x,Weights,bias,dropout,fla):
    
    ## Convolutional layer 1(卷积层1)
    conv1 = conv2d(x,Weights['con1_w'],bias['conv1_b']) ##100*100*64
    conv1 = maxpool2d(conv1,2) ##经过池化层1 shape：50*50*64
     
    ## Convolutional layer 2（卷积层2）
    conv2 = conv2d(conv1,Weights['con2_w'],bias['conv2_b']) ##50*50*128
    conv2 = maxpool2d(conv2,2) ##经过池化层2 shape:25*25*128
    ## Fully connected layer 1(全连接层1)
    flatten = tf.reshape(conv2,[-1,fla]) ##Flatten层，扁平化处理
    fc1 = tf.add(tf.matmul(flatten,Weights['fc_w1']),bias['fc_b1'])
    fc1 = tf.nn.relu(fc1) ##经过relu激活函数
#    print(flatten.get_shape())
    ## Fully connected layer 2(全连接层2)
    fc2 = tf.add(tf.matmul(fc1,Weights['fc_w2']),bias['fc_b2']) ##计算公式：输出参数=输入参数*权值+偏置
    fc2 = tf.nn.relu(fc2) ##经过relu激活函数
    
    ## Dropout（Dropout层防止预测数据过拟合）
    fc2 = tf.nn.dropout(fc2,dropout)
    ## Output class prediction
    prediction = tf.add(tf.matmul(fc2,Weights['out']),bias['out']) ##输出预测参数
    return prediction

def CNN_init():
    
    images_test_20,labels_test_20=load_small_data('./Test/',10,False)
    test_x,test_y=prepare_data(images_test_20,labels_test_20,10)
    n_classes=10 ##数据的类别数
    kernel_h=kernel_w=5 ##卷积核尺寸
    depth_in=3 ##图片的通道数
    depth_out1=64 ##第一层卷积的卷积核个数
    depth_out2=128 ##第二层卷积的卷积核个数image_size=train_x.shape[1] ##图片尺寸           
    image_size=test_x.shape[1]
    fla=int((image_size*image_size/16)*depth_out2) ##用于扁平化处理的参数经过两层卷积池化后的图像大小*第二层的卷积核个数
    ckpt_file_path = "./CNNData/save_net.ckpt"
    
    tf.reset_default_graph()
    
    Weights={"con1_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_in,depth_out1]), name = "con1_w"),
            "con2_w":tf.Variable(tf.random_normal([kernel_h,kernel_w,depth_out1,depth_out2]), name = "con2_w"),
            "fc_w1":tf.Variable(tf.random_normal([int((image_size*image_size/16)*depth_out2),1024]), name = "fc_w1"),
            "fc_w2":tf.Variable(tf.random_normal([1024,512]), name = "fc_w2"),
            "out":tf.Variable(tf.random_normal([512,n_classes]), name = "out1")}
    
    ##定义各卷积层和全连接层的偏置变量
    bias={"conv1_b":tf.Variable(tf.random_normal([depth_out1]), name = "conv1_b"),
          "conv2_b":tf.Variable(tf.random_normal([depth_out2]), name = "conv2_b"),
          "fc_b1":tf.Variable(tf.random_normal([1024]), name = "fc_b1"),
          "fc_b2":tf.Variable(tf.random_normal([512]), name = "fc_b2"),
          "out":tf.Variable(tf.random_normal([n_classes]), name = "out2")}
    
    
    saver = tf.train.Saver()
    
    sess = tf.Session() 
    saver.restore(sess, ckpt_file_path)
    sess.run(Weights)
    sess.run(bias)
    
    x=tf.placeholder(tf.float32,[None,100,100,3]) 
    keep_prob=tf.placeholder(tf.float32) ##dropout的placeholder(解决过拟合)
    prediction=conv_net(x,Weights,bias,keep_prob,fla) ##生成卷积神经网络
    dict_data = {'x':x,'keep_prob':keep_prob,'prediction':prediction, \
                 'Weights':Weights, 'bias':bias, "n_classes":n_classes}
    return dict_data,sess
    
def run_CNN(x,keep_prob,prediction,img,sess):
    
    test_feed={x:img,keep_prob: 0.8} 
    y1 = sess.run(prediction,feed_dict=test_feed)
    test_classes = np.argmax(y1,1)
    lab=os.listdir("./Test")
    return lab[test_classes[0]]

def text_Accuracy(x,keep_prob,prediction,sess,n_classes):  
    
    y=tf.placeholder(tf.float32,[None,n_classes]) ##feed到神经网络的标签数据的类型和shape
    images_test_20,labels_test_20=load_small_data('./Test/',n_classes,False)
    test_x,test_y=prepare_data(images_test_20,labels_test_20,n_classes)
    ## 评估模型
    correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
    test_x=test_x[0:400]
    test_y=test_y[0:400]
    test_feed={x:test_x,y:test_y,keep_prob: 0.8} 
    # y1 = sess.run(prediction,feed_dict=test_feed)
    # test_classes = np.argmax(y1,1)
    # class_label(test_classes, test_x)
    data = sess.run(accuracy,feed_dict=test_feed)
    print('Testing Accuracy:',data) 
    
    return data

def batch_process(x,keep_prob,prediction,sess,img_set):

    img_set = np.array(img_set)
    test_feed={x:img_set,keep_prob: 0.8} 
    y1 = sess.run(prediction,feed_dict=test_feed)
    test_classes = np.argmax(y1,1)
    lab=np.array(os.listdir("./Test"))
    return lab[test_classes]

if __name__ == '__main__':
    dict_data,sess= CNN_init()
    x1 = dict_data['x']
    keep_prob = dict_data['keep_prob']
    prediction = dict_data['prediction']
    n_classes = dict_data['n_classes']
    text_Accuracy(x1,keep_prob,prediction,sess,n_classes)    


