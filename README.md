# zhihu_kanshan_cup_2017
2017知乎看山杯比赛，我的部分代码。仅提供了单模型最高分的RCNN+ATTENTION模型。

详情请移步至我的博文：[大规模文本分类实践-知乎看山杯总结]()

# 数据下载与说明
数据存在百度云： http://pan.baidu.com/s/1bpnNRQJ
数据说明：[移步至此]()

# 运行环境：
- python版本：py3
- keras版本：Keras (2.0.6)  其中prozhuchen修改了keras的training.py 文件，使得其能够处理稀疏的label，具体修改：training.py  第375~455行。
具体原理是在每一个batch时把稀疏的label转化为dense的，省空间。您也可以使用keras的fit_on_batch函数，但是如果您使用fit函数的话，必须使用我们修改后的源代码文件。
- 本代码提供了修改后的training.py文件，在modified_keras_files中     请将其替换您keras目录下的：keras/engine/training.py文件

# 执行方法
## 训练：
python3 main.py train

## 预测：
python3 main.py pred


