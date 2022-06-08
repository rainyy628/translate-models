#get acc of model'applications on text 
import pandas as pd
import re
file_path1='/root/code/yaoyao/mb/biaozhu/大学_注释.csv'
file_path2='/root/code/yaoyao/mb/guwen_zhushi/大学_6sentences.csv'
df_tagging=pd.read_csv(file_path1)
#,sentences,words,meanings,labels
df_model=pd.read_csv(file_path2)
#sentence,words,words_meaning,second_meaning 
df_model.rename(columns={'sentence':'sentences','words':'words'},inplace=True)

#join df_tagging and df_model together
df_all=df_tagging.merge(df_model,on=['sentences','words'])
df_all.to_csv('/root/code/yaoyao/mb/guwen_zhushi/guwen_大学_con_no_w_guwen.csv',index=False)
# df_all.to_csv('/root/code/yaoyao/mb/biaozhu/daxue_model_acc.csv')
xc_dict='/root/code/yaoyao/mb/data/data_3_31/xc_dict.csv'
df_xc=pd.read_csv(xc_dict)
#,word,meaning,examples
xc_list=df_xc['word'].tolist()
xc_true=0
xc_all=0
all,true=0,0
sentences=[]
words=[]
word_meanings=[]
temp_meaning=[]
trues=[]
second_trues=[]
zero=0#记录有多少没有选项
for i in range(len(df_all)):
    all+=1
    if str(df_all.iloc[i]['labels'])=='0':
        zero+=1
    if '0' in str(df_all.iloc[i]['labels']):
        predict_num=re.sub('0','',str(df_all.iloc[i]['labels']))
        predict_num=re.sub(' ','',predict_num)
    w=df_all.iloc[i]['words']
    if w in xc_list:
        xc_all+=1
    predict_meaning=df_all.iloc[i]['words_meaning']
    predict_meaning_alternate=df_all.iloc[i]['second_choice']
    predict_meaning=re.sub(' ','',predict_meaning)
    predict_meaning_alternate=re.sub(' ','',predict_meaning_alternate)
    predict_num=str(df_all.iloc[i]['labels'])
    all_meanings=df_all.iloc[i]['meanings']
    all_meanings=re.sub('<例句：.*>','',all_meanings)
    all_meanings=all_meanings.split('\r\n')
    c_time=0
    for meanings in all_meanings:
        if predict_num in meanings:
            if predict_meaning in meanings:
                c_time += 1
                if c_time>1:
                    print('---------------')
                    print(predict_num)
                    print(predict_meaning)
                    print(all_meanings)
                true+=1
                if w in xc_list:
                    xc_true+=1
                words.append(w)
                temp_meaning.append(predict_meaning)
                sentences.append(df_all.iloc[i]['sentences'])
                word_meanings.append(df_all.iloc[i]['meanings'])
                trues.append(1)
                second_trues.append(0)
            elif predict_meaning_alternate in meanings:
                true+=1
                if w in xc_list:
                    xc_true+=1 
                words.append(w)
                temp_meaning.append(predict_meaning_alternate+'['+predict_meaning_alternate+']')
                sentences.append(df_all.iloc[i]['sentences'])
                word_meanings.append(df_all.iloc[i]['meanings'])
                trues.append(0)
                second_trues.append(1)
            break

print('xc:{}/{}'.format(xc_true,xc_all))
print('acc:{}/{}'.format(true,all))
print('zeros:{}/{}'.format(zero,all))
df_new=pd.DataFrame({'sentences':sentences,'words':words,'word_meanings':word_meanings,'predict_meaning':temp_meaning,'trues':trues,'second_trues':second_trues})
#df_new.to_csv('/root/code/yaoyao/mb/biaozhu/daxue_model_true_top2.csv')