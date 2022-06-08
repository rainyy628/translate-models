import pandas as pd
import re
file_origin='/root/code/yaoyao/mb/biaozhu/中庸2_all.csv'
file_verify_0='/root/code/yaoyao/mb/biaozhu/中庸验证2_1_已完成.csv'
file_verify_1='/root/code/yaoyao/mb/biaozhu/中庸验证2_2_已完成.csv'
file_verify_2='/root/code/yaoyao/mb/biaozhu/中庸验证2_3_已完成.csv'
# file_verify_1='/root/code/yaoyao/mb/biaozhu/中庸_标注验证1（完成版）.csv'
#句子,字,释义,正确选项（填序号）
df_origin=pd.read_csv(file_origin)
df_verify_0=pd.read_csv(file_verify_0,encoding='GBK')
df_verify_1=pd.read_csv(file_verify_1,encoding='GBK')
df_verify_2=pd.read_csv(file_verify_2,encoding='GBK')

# df_verify=df_verify.dropna(subset=['句子'],how='all')
df_verify=df_verify_0.append(df_verify_1)
df_verify=df_verify.append(df_verify_2)

s_list=df_origin['句子'].tolist()
all=0
true=0
sentences=[]
words=[]
meanings=[]
lables0=[]
lables1=[]
#句子,字,释义,正确选项（填序号）：
for i in range(len(df_verify)):
    s=df_verify.iloc[i]['句子']
    # s=re.sub('\r','',s)
    s_index=s_list.index(s)
    all+=1
    choice_verify=df_verify.iloc[i]['正确选项（填序号）：']
    choice_origin=df_origin.iloc[s_index]['正确选项（填序号）：']
    if type(choice_verify)==int:
        choice_verify=str(choice_verify)
    if choice_verify==choice_origin:
        true+=1
    else:
        #将lable统一为字符串
        print(choice_verify)
        print(type(choice_verify))
        print(choice_origin)
        print(type(choice_origin))
        print('--------------------')
        words.append(df_origin.iloc[s_index]['字'])
        meanings.append(df_origin.iloc[s_index]['释义'])
        lables0.append(df_origin.iloc[s_index]['正确选项（填序号）：'])
        lables1.append(df_verify.iloc[i]['正确选项（填序号）：'])

print('{}/{}'.format(true,all))     
df_fault=pd.DataFrame({'句子':sentences,'字':words,'释义':meanings,'标注选项（填序号）：':lables0,'验证选项（填序号）：':lables1})
df_fault.to_csv('/root/code/yaoyao/mb/biaozhu/fault_infos_1.csv')
