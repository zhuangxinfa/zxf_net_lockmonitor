from test_accuracy import get_one_image
from test_accuracy import evaluate_one_image
from input_data import get_files
import os
import numpy as np
if __name__ == '__main__':  
    path=os.path.abspath('.')
    testdir = path+'/pic/test_train' #存储测试图片的目录
    train1,train_label2,val2,val_label2 = get_files(testdir,1.0)
    n = len(val2)  #n是整形正好是16个
    ind = np.random.randint(0, n) 
    img = get_one_image(val2,ind)  #通过改变参数train or val，进而验证训练集或测试集  
    evaluate_one_image(img)