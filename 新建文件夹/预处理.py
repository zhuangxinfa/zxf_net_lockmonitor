#将原始图片转换成需要的大小，并将其保存  
#制作threcords然乎从tfrecords中读取数据，reshape成64x64并保存
#========================================================================================  
import os  
import numpy as np  
import tensorflow as tf    
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
    
#原始图片的存储位置  
path=os.path.abspath('.')
orig_picture =path + '/pic'#图片在pic文件夹下的 mao和gou 的文件夹下  
#orig_picture = 'C:/Users/zhuan/Desktop/tf/pic'  
  
#生成图片的存储位置  
gen_picture =path+ '/pic/test_data'  #会在inputdata下生产64x64的数据
  
#需要的识别类型  
classes = {'ok','liefeng','qinshituoluo','wailu','wusun'}   #这是一个dict只不过他的value为零
  
#样本总数  
#num_samples = 74   

num_samples = 5702
     
#制作TFRecords数据    
def create_record():    
    writer = tf.python_io.TFRecordWriter("train.tfrecords")    
    for index, name in enumerate(classes): #enumerate既可以遍历又可以索引，index是索引号，name是字典里的key值   
        class_path = orig_picture +"/"+ name+"/"  
        i = 0
        for img_name in os.listdir(class_path):   
            img_path = class_path + img_name    
            img = Image.open(img_path)   
            img = img.resize((64, 64)) 
            #设置需要转换的图片大小
            i = i+1
            print("成功转化",i,"张图片")
#            img = img.tf.image.resize_images()   #设置需要转换的图片大小  
            img_raw = img.tobytes()      #将图片转化为原生bytes    
#            print (index,img_raw)    
            example = tf.train.Example(    
               features=tf.train.Features(feature={    
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),    
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))    
               }))    
			    # 写 TFRecord
            writer.write(example.SerializeToString())    
    writer.close()    
#      
#=======================================================================================  
def read_and_decode(filename):    
    # 创建文件队列,不限读取的数量    
    filename_queue = tf.train.string_input_producer([filename])    
    # create a reader from file queue    
    reader = tf.TFRecordReader()    
    # reader从文件队列中读入一个序列化的样本    
    _, serialized_example = reader.read(filename_queue)    
    # get feature from serialized example    
    # 解析符号化的样本    
    features = tf.parse_single_example(    
        serialized_example,    
        features={    
            'label': tf.FixedLenFeature([], tf.int64),    
            'img_raw': tf.FixedLenFeature([], tf.string)    
        })    
    label = features['label']    
    img = features['img_raw']    
    img = tf.decode_raw(img, tf.uint8)    
    img = tf.reshape(img, [64, 64, 3])    
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5    
    label = tf.cast(label, tf.int32)    
    return img, label    
  
#=======================================================================================  
if __name__ == '__main__':    
    create_record()    
    batch = read_and_decode('train.tfrecords')    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())    
        
    with tf.Session() as sess: #开始一个会话      
        sess.run(init_op)      
        coord=tf.train.Coordinator()      
        threads= tf.train.start_queue_runners(coord=coord)  
#        print(sess.run(batch))
        for i in range(num_samples):    
            try:
                example, lab = sess.run(batch) #在会话中取出image和label  
#            print(example)
            except:
                print ("错误异常")
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的   
            img.save(gen_picture+'/'+str(lab)+'/'+str(i)+'samples'+str(lab)+'.jpg')
            print("第",i+1,"张图片成功输出")
            #存下图片;注意cwd后边加上‘/’      
#            print(example, lab)      
        coord.request_stop()      
        coord.join(threads)     
        sess.close()