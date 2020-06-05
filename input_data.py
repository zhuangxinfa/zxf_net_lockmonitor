#-----------------生成图片路径和标签的List------------------------------------  
import math  
import numpy as np  
import tensorflow as tf  
#import matplotlib.pyplot as plt  
import os
path=os.path.abspath('.')
train_dir = path+'/pic/test_data'  

#train_dir = 'C:/Users/zhuan/Desktop/tf/pic/input_data'  
  
sample0 = []  
label_0 = []  
sample1 = []  
label_1 = []  
sample2 = []  
label_2 = []  
sample3 = []  
label_3 = []  


#对应的列表中，同时贴上标签，存放到label列表中。  
def get_files(file_dir, ratio):  

    for file in os.listdir(file_dir+'/0'):  
        sample0.append(file_dir +'/0'+'/'+ file)    
        label_0.append(0)  #ac
    for file in os.listdir(file_dir+'/1'):  
        sample1.append(file_dir +'/1'+'/'+file)  
        label_1.append(1)  #ao
    for file in os.listdir(file_dir+'/2'):  
        sample2.append(file_dir +'/2'+'/'+ file)    
        label_2.append(2)  #bc
    for file in os.listdir(file_dir+'/3'):  
        sample3.append(file_dir +'/3'+'/'+file)  
        label_3.append(3)#bo
   
#step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab） 
    image_list = np.hstack((sample0, sample1,sample2,sample3))      #它其实就是水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反。
#    print(image_list)
    #image_list = np.hstack((husky, jiwawa, poodle, qiutian))  
    label_list = np.hstack((label_0, label_1,label_2,label_3))  #它其实就是水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反。
#    print (label_list)
    #利用shuffle打乱顺序  
    temp = np.array([image_list,
                     label_list])  
#    print(temp)
    temp = temp.transpose()  #转置
    np.random.shuffle(temp)  #打乱了之后还是对应的
#    print(temp)#temp是一个二维数组 有280行 两列 ，第一列是图片第二列是标签
      
    #从打乱的temp中再取出list（img和lab）  
#    image_list = list(temp[:, 0])  
#    label_list = list(temp[:, 1])  
#    label_list = [int(i) for i in label_list]  
#    return image_list, label_list  
#      
    #将所有的img和lab转换成list  
    all_image_list = list(temp[:, 0])  
#    print (all_image_list)
    #X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，
   #第二维中取第0个数据，直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的
    #第1个数据。
    all_label_list = list(temp[:, 1])  
#    print(all_label_list)
  
    #将所得List分为两部分，一部分用来训练tra，一部分用来测试val  
    #ratio是测试集的比例  
    n_sample = len(all_label_list)  
#    print (n_sample) #输出280
    n_val = int(math.ceil(n_sample*ratio))   #测试样本数  
    n_train = n_sample - n_val   #训练样本数  
  
    tra_images = all_image_list[0:n_train]  
    tra_labels = all_label_list[0:n_train]  
    tra_labels = [int(float(i)) for i in tra_labels]  
    val_images = all_image_list[n_train:-1]  
    val_labels = all_label_list[n_train:-1]  
    val_labels = [int(float(i)) for i in val_labels]  
  
    return tra_images, tra_labels, val_images, val_labels  #返回的都是list
      
      
#---------------------------------------------------------------------------  
#--------------------生成Batch----------------------------------------------  
  
#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab  
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像  
#   image_W, image_H, ：设置好固定的图像高度和宽度  
#   设置batch_size：每个batch要放多少张图片  
#   capacity：一个队列最大多少  
def get_batch(image, label, image_W, image_H, batch_size, capacity):  
    #转换类型  
    image = tf.cast(image, tf.string)  #转换成字符串类型数据本来是Python的list类型，转换成tensorflow能够识别的tf.string
    label = tf.cast(label, tf.int32)  #转换成int类型
  
    # make an input queue  
    input_queue = tf.train.slice_input_producer([image, label])  
  
    label = input_queue[1]  
    image_contents = tf.read_file(input_queue[0]) #read img from a queue    
      
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用 png等。  
    image = tf.image.decode_jpeg(image_contents, channels=3)   
      
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。  
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)  
    image = tf.image.per_image_standardization(image)  
  
#step4：生成batch  
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32   
#label_batch: 1D tensor [batch_size], dtype=tf.int32  
    image_batch, label_batch = tf.train.batch([image, label],  
                                                batch_size= batch_size, #batch大小是20 
                                                num_threads= 32,   
                                                capacity = capacity) # capacity是两百 是队列容量
    #重新排列label，行数为[batch_size]  
    label_batch = tf.reshape(label_batch, [batch_size])  
    image_batch = tf.cast(image_batch, tf.float32)  
    return image_batch, label_batch    #返回两个batch