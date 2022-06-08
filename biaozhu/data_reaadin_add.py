#把标注的文件读入
import pandas as pd
import numpy as np

folder_name='/root/code/yaoyao/mb/biaozhu'
file1='/中庸2_1待注释.csv'
file2='/中庸2_2待注释.csv'
file3='/中庸2_3待注释.csv'

df1=pd.read_csv(folder_name+file1)
print(len(df1))
df2=pd.read_csv(folder_name+file2)
print(len(df2))
df3=pd.read_csv(folder_name+file3)
print(len(df3))

df1=df1.append(df2,ignore_index=True)
df1=df1.append(df3,ignore_index=True)
print(len(df1))
# df1.to_csv(folder_name+'/中庸2_all.csv')

# df2=df2.dropna(subset=['句子','正确选项（填序号）：'],how='all')
# sentences1=df1['句子'].tolist()
# sentences2=df2['句子'].tolist()
# words=df1['字'].tolist()
# meanings=df1['释义'].tolist()
# choice=df2['正确选项（填序号）：'].tolist()

# labels=[]
# #将新标的数据放入原数据格式
# for i in range(len(sentences1)):
#     if sentences2.count(sentences1[i])>0:
#         j=sentences2.index(sentences1[i])
#         labels.append(choice[j])
#     else:
#         #没有该句子
#         labels.append(0)
    

# df_new=pd.DataFrame({'sentences':sentences1,'words':words,'meanings':meanings,'labels':labels})
# df_new.to_csv(folder_name+'/中庸2_注释.csv')