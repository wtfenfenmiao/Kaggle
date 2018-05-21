#coding:utf-8
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# PassengerId      int64    不要
# Survived         int64    这个是Y
# Pclass           int64    分了三类
# Name            object    这个也不要
# Sex             object    两类
# Age            float64    算是连续特征吧
# SibSp            int64    算成类别
# Parch            int64    算成类别
# Ticket          object    这个也不要
# Fare           float64    连续特征
# Cabin           object    这个做成离散特征吧，就是数有几个cabin，Nan就是-1
# Embarked        object    离散特征

#train和test放在一起玩
#缺失值处理，离散的补成-1，连续的补成平均数

#二分类问题，有连续特征，还是先用lightgbm，要用上交叉验证


#看一下数据什么样的
def data_view():
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    columns=list(train.columns)
    datatypes=train.dtypes
    print (columns)
    print (datatypes)
    print (len(train))    #891行
    print (len(test))     #418行

    #print (train.iloc[0:50,:])
    #print (test.iloc[0:50,:])


def data_process():
    print ("hello")
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")

    #标识一下，Y是-1的都是test，是01的都是train
    test["Survived"]=-1

    #把train和test合起来，然后index忽略，因为原来的index没用了
    data=pd.concat([train,test],ignore_index=True)

    #Age字段处理，缺失值填充成均值
    #print (data["Age"].mean())
    data["Age"]=data["Age"].fillna(data["Age"].mean())



    #Cabin字段处理，数有多少客舱，没有的填充成-1
    #这个我也不知道怎么处理，这个东西是啥我不太理解，查着是“客舱号码”，可能，数字越多的越有钱？？？所以我数了有多少个
    #找Cabin的缺失值，np.isnan没办法用，因为np.isnan只能用于数值型与np.nan组成的numpy数组。所以就先填成-1了，反而省事了
    #print (data["Cabin"][0])
    #print (type(data["Cabin"][0]))
    data["Cabin"]=data["Cabin"].fillna(-1)
    #print (data.loc[data["Cabin"]!=-1,"Cabin"].apply(lambda x:len(x.split())))
    data.loc[data["Cabin"] != -1, "Cabin"]=data.loc[data["Cabin"]!=-1,"Cabin"].apply(lambda x:len(x.split()))

    #Fare字段处理，缺失值填充成平均数
    data["Fare"]=data["Fare"].fillna(data["Fare"].mean())

    #name不要
    #ticket不要
    data.drop(['Name','Ticket'],axis=1,inplace=True)



    #要做one-hot的
    oneHot_feature=["Embarked","Parch","Pclass","Sex","SibSp","Cabin"]

    le=LabelEncoder()     #这个得有，下面直接用LabelEncoder的话编不过
    for feature in oneHot_feature:
        data[feature].fillna(-1)
        data[feature]=le.fit_transform(list(data[feature].values))


    #one-hot会变成稀疏矩阵，所以这里train和test先准备出来
    train=data[data.Survived!=-1]
    train_y=train["Survived"]
    train.drop("Survived",axis=1,inplace=True)
    test=data[data.Survived==-1]
    test.drop("Survived",axis=1,inplace=True)
    test_passenger=test[["PassengerId"]]
    train_x=train[["Age","Fare"]]
    test_x=test[["Age","Fare"]]


    #print (data)
    enc=OneHotEncoder()
    for feature in oneHot_feature:
        enc.fit(data[feature].values.reshape(-1,1))
        train_a=enc.transform(train[feature].values.reshape(-1,1))
        test_a=enc.transform(test[feature].values.reshape(-1,1))
        train_x=sparse.hstack((train_x,train_a))
        test_x=sparse.hstack((test_x,test_a))

    return train_x,train_y,test_x,test_passenger



def lgb_pre(train_x,train_y,test_x,test_passenger):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=0)

    print("start grid search!")

    # 查查默认参数都是啥，在附近改。没出现的就都用默认参数,想调谁写谁就行，不至于全写上
    param_test = {
        'num_leaves': range(5, 80, 5),
        'learning_rate': [0.05, 0.15, 0.01],
        'n_estimators': range(10, 100, 10)
    }

    estimator = LGBMClassifier(
        objective='binary',
        silent=True
    )
    # 指标用f1，是一个二分类的评价指标
    gsearch = GridSearchCV(estimator, param_grid=param_test, scoring='f1', cv=3)
    gsearch.fit(train_x, train_y)
    params = gsearch.best_params_
    model_lgb = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=params['num_leaves'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        silent=True
    )

    print('Start training...')
    model_lgb.fit(train_x, train_y, eval_set=[(val_x, val_y)])

    print('Start predicting...')
    out = test_passenger
    out["Y"] = model_lgb.predict(test_x)
    out.to_csv("submission.csv", index=False, header=["PassengerId", "Survived"])
    print (params['num_leaves'])
    print(params['learning_rate'])
    print(params['n_estimators'])
    print ("finish!!")

def call_lgb():
        train_x, train_y, test_x, test_passenger=data_process()
        lgb_pre(train_x, train_y, test_x, test_passenger)











if __name__=="__main__":
    print ("hello")
    #data_view()
    #data_process()
    call_lgb()
