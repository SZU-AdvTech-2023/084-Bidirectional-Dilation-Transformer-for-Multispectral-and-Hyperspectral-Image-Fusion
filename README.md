# Bidirectional Dilation Transformer For Multispectral And Hyperspectral Image Fusion

本实验是在原论文代码基础上进行了一定的修改。[源代码地址](https:\\github.com\Dengshangqi\BDT)

## Platform

+ windows 10
+ pytcharm2023
+ python 3.8.5

## 修改

在高光谱的空间信息读取模块 D-spa 上增加了通道分割的操作将输入的图片在光谱维度进行扩大，然后分成两块分别进行窗口注意力机制提取空间信息，再将得到的结果相加，扩大了对于光谱通道信息的感受野。在作者给出的 cave 和 harvard 数据集上面重新训练。然后由于在原论文之中作者只描述了在训练集和验证集上面进行模型的效果展示，本人编写了一个测试文件，用来在测试集上运行来观察模型效果。

## 操作说明

将高光谱数据集放入 data 文件之中，并在 args_parser.py 文件中配置好对应的训练集、验证集和测试集的路径，运行 main.py 文件进行模型训练，训练过程中会在 checkpoints 文件加下面生成相应的最优模型参数，方便后面跑测试集的时候载入使用。运行 test.py 文件进行测试，会生成参考图像、融合结果图、结果差异图和模型光谱折线图在 data 文件夹下面。