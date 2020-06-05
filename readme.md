
## 用户训练门锁监控tensorflow模型的python代码

![Pandao editor.md](./comment.bmp "Pandao editor.md")

# 目录说明
- pic文件夹下存放的是要训练的图片 每一类图片存储在一个文件夹里边，有几类就在pic文件夹下有几个对应的文件夹
- input_data.py用于读取图片并进行预处理
- model.py定义了网络模型 就是一个最简单不过的cnn卷积神经网络
- train.py可以训练模型 在里边可以调节训练时的一些参数