#创建triplet dataset
import math
import random

import nn
import numpy as np
from sentence_transformers import models, SentenceTransformer

model_name='imxly/sentence_rtb3'
word_embedding_model=models.Transformer(model_name)
pooling_model=models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                             pooling_mode_mean_tokens=True,
                             pooling_mode_cls_token=False,
                             pooling_mode_max_tokens=False)
model=SentenceTransformer(modules=[word_embedding_model,pooling_model])

def evaluate(model,s1,s2):
    #计算余弦相似度
    import numpy as np
    v1=model.encode(s1)
    v2=model.encode(s2)
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    return v1.dot(v2)

#先导入初始数据
import pandas as pd
file_name='/root/code/yaoyao/mb/data_origin/zhushi_data3.csv'
dataset=pd.read_csv(file_name)
data_origin=dataset.iloc[:,1:4]
dataset_loc0='/root/code/yaoyao/mb/data'
#创建anchor, positve, negtive三个list
#list_item={word，sentence}
anchor=[]
positive=[]
negtive=[]
anchor_list=[]
#anchor_list用于判断是否使用过该数据当做anchor
#创建记录是否计算过word_sim的矩阵
length=len(data_origin)
s=(length,length)
sim_flag=np.zeros(s)
#sim_flag中为可能相似，为1为确定不相似

data_num=len(data_origin)
data_list=data_origin.values.tolist()
#打乱原有数据顺序
random.shuffle(data_list)
# for i in range(10):
#     print(random.choice(data_list))

#记录是否已经输出过一次
flag_out=0
import random
#positive是全list筛查, 再随机筛查相同个数的negtive
for i in data_list:
    if i not in anchor_list:
        anchor_list.append(i)
        i_index=data_list.index(i)
        #记录临时的negtive备选项
        temp_negtive=[]
        #记录找到的positive数量
        temp_time=0
        #找positive
        for k in range(1000):
            j=random.choice(data_list)
            if i[0]!=j[0] or i[2]!=j[2]:
                word_sim=evaluate(model,i[1],j[1])
                if word_sim>0.62:
                    positive.append([j[0],j[2]])
                    anchor.append([i[0],i[2]])
                    temp_time+=1
                else:
                    temp_negtive.append([j[0],j[2]])
        #补齐negtive
        neg_num=len(temp_negtive)
        #是否值太少，需要增加
        if temp_num==0:
            for k in range(2000):
            j=random.choice(data_list)
            if i[0]!=j[0] or i[2]!=j[2]:
                word_sim=evaluate(model,i[1],j[1])
                if word_sim>0.62:
                    positive.append([j[0],j[2]])
                    anchor.append([i[0],i[2]])
                    temp_time+=1
                else:
                    temp_negtive.append([j[0],j[2]])
        for k in range(temp_time):
            negtive.append(temp_negtive[k%neg_num])
        
        #顺次找positve
        # for j in data_list:
        #     j_index=data_list.index(j)
        #     #判断anchor和positive是不相同的
        #     if i[0]!=j[0] or i[2]!=j[2]:
        #         #判断两者是有机会相似的
        #         if sim_flag[i_index][j_index]==0:
        #             word_sim=evaluate(model,i[1],j[1])
        #             if word_sim>0.62:
        #                 #此时为正例
        #                 positive.append([j[0],j[2]])
        #                 print(str(i[1])+" "+str(j[1])+" word_sim:"+str(word_sim))
        #             else:
        #                 #若不相似，则记住该组不相似例子，纺织重复计算
        #                 sim_flag[i_index][j_index]=1
        #                 sim_flag[j_index][i_index]=1
        # #找完所有positve就补充anchor和negtive
        # #补充anchor
        # #补充anchor
        # for temp_count in range(len(anchor),len(positive)):
        #     anchor.append([i[0],i[2]])
        # #补充negitve
        # while len(positive)>len(negtive):
        #     temp_item=random.choice(data_list)
        #     word_sim=evaluate(model,temp_item[1],i[1])
        #     if word_sim<=0.62:
        #         negtive.append([temp_item[0],temp_item[2]])
        print("time:{}/{}".format(data_list.index(i),length))
        print("anchor_len:"+str(len(anchor))+" positve_len:"+str(len(positive))+" negtive_len:"+str(len(negtive)))
        


#将三个列表合成dataframe存储
data_df=pd.DataFrame({'anchor':anchor,'positive':positive,'negtive':negtive})

#划分train dev test数据集
from sklearn.model_selection import train_test_split

split_rate=0.2
if len(data_df) > 100000:
    split_rate=0.1
elif len(data_df) > 200000:
    split_rate=0.06

train_set,dev_test=train_test_split(data_df,test_size=split_rate,random_state=0)
dev_set,test_set= train_test_split(dev_test,test_size=0.5,random_state=1)
train_set.to_csv(dataset_loc0+'/train.csv')
dev_set.to_csv(dataset_loc0+'/dev.csv')
test_set.to_csv(dataset_loc0+'/test.csv')