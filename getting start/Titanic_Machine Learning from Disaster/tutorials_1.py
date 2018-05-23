#coding:utf-8

import pandas as pd

#读取训练数据和测试数据
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
combine=[train_df,test_df]     #这还是两个，这两个没有任何往一起放的操作
#print (combine)    #这一行要老命，数据量大可不能这么玩，可以打印50行看看啥的

#首先，看看都有啥特征，除了从它给的数据介绍里看，我们还可以看看列名都是啥
print (train_df.columns.values)

