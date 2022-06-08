#获取展示的例子
import pandas as pd
import re
filename='/root/code/yaoyao/mb/biaozhu/大学_注释.csv'
#,sentences,words,meanings,labels
df=pd.read_csv(filename)
str='所谓平天下在治其国者，上老老而民兴孝，上长长而民兴弟，上恤孤而民不倍，是以君子有絜矩之道也'

for i in range(len(df)):
    temp_s=df.iloc[i]['sentences']
    temp_s=re.sub('[【,】]','',temp_s)
    if temp_s==str:
        m=df.iloc[i]['meanings']
        l=df.iloc[i]['labels']
        m=re.sub('<例句：.*>','',m)
        m=m.split('\r\n')
        m_item="none"
        for m_item in m:
            if l in m_item:
                true_m=m_item
        msg='sentence:{}  word:{}  meanings:{}'
        print(msg.format(df.iloc[i]['sentences'],df.iloc[i]['words'],true_m))
        print('\n')