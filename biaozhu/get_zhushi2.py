#第一次标注标注为0或标注掉的内容
import pandas as pd
import numpy as np

file_1='/root/code/yaoyao/mb/biaozhu/中庸2_注释.csv'
df=pd.read_csv(file_1)
sentences=[]
words=[]
meanings=[]
labels=[]

for i in range(len(df)):
    label=df.iloc[i]['labels']
    if type(label)==type('98'):
        if label=='0' or ' 0' in label:
            words.append(df.iloc[i]['words'])
            sentences.append(df.iloc[i]['sentences'])
            meanings.append(df.iloc[i]['meanings'])
            labels.append(df.iloc[i]['labels'])
    else:
        words.append(df.iloc[i]['words'])
        sentences.append(df.iloc[i]['sentences'])
        meanings.append(df.iloc[i]['meanings'])
        labels.append(df.iloc[i]['labels'])

df_new=pd.DataFrame({'sentences':sentences,'words':words,'meanings':meanings,'labels':labels})
df_new.to_csv('/root/code/yaoyao/mb/biaozhu/中庸2_再注释.csv')