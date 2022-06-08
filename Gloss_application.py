from jiayan import load_lm
from jiayan import CRFSentencizer,CharHMMTokenizer
import GlossBert as myModel
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import re
import random
import copy

PAD, CLS,SEP='[PAD]','[CLS]','[SEP]'
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")

model = AutoModel.from_pretrained("ethanyt/guwenbert-base")

#读取字典
df_dict=pd.read_csv('/root/code/yaoyao/mb/data_origin/yao_dict_xc_amend.csv')
#words,cixing,pinyin,words_meaning,sentences
#获取字列表
word_dict=df_dict['words'].tolist()

lm=load_lm('jiayan.klm')
sentencizer=CRFSentencizer(lm)
sentencizer.load('cut_model')

#分词
tokenizer=CharHMMTokenizer(lm)
#list(tokenizer.tokenize(text)

#断句/根据句号分句子
def passage_seg(str):
    #判断是否有句读
    judou_ch='。'
    if str.find(judou_ch)<0:
        #进行断句
        str_list=sentencizer.sentencize(str)
    else:
        str_list=str.split('。')
    return str_list


#导入训练的模型Contrastive bert
project_dict="/root/code/yaoyao/mb"
model_name='GlossBert'
config=myModel.Config(project_dict)

glossbert_model=myModel.Model(config).to(config.device)
glossbert_model.load_state_dict(torch.load(config.save_path))
c_co=glossbert_model.state_dict()


glossbert_model.eval()

#sentence,w:meaning
def model_out(patch):
    s_encoded=get_encoded(patch)
    tokens_id=s_encoded[0]
    seq_len=s_encoded[1]
    mask=s_encoded[2]
    tokens_id=torch.LongTensor([tokens_id]).to(torch.device('cuda'))
    seq_len=torch.LongTensor([seq_len]).to(torch.device('cuda'))
    mask=torch.LongTensor([mask]).to(torch.device('cuda'))
    s_encoded=[tokens_id,seq_len,mask]
    with torch.no_grad():
        output=glossbert_model(s_encoded)
    return output


def get_encoded(patch):
    s1_tokens=config.tokenizer.tokenize(patch[0])
    s2_tokens=config.tokenizer.tokenize(patch[1])
    tokens=[CLS]+s1_tokens+[SEP]+s2_tokens
    seq_len=len(tokens)
    tokens_id=config.tokenizer.convert_tokens_to_ids(tokens)
    mask=[1]*len(tokens_id)
    return tokens_id,seq_len,mask
    

def get_s_split(str):
    str=re.sub('[\[,\],\']','',str)
    strs=str.split()
    return strs

print(model_out(['然而灵用不同，“玄”化各异', '玄：黑中带红的颜色']))
#将一段文章进行自动注释
def get_noted(str_list):
    word_list=[]
    word_meaning_list=[]
    second_meaning_list=[]
    sentence_list=[]
    str_nums=len(str_list)
    for str_index in range(str_nums):
        str=str_list[str_index]
        str=re.sub('\u3000','',str)
        words=[]
        meanings=[]
        second_meanings=[]
        sentences=[]
        #获取更长长度的句子
        long_str=copy.deepcopy(str)
        str_before=''
        str_behind=''
        # if str_index-1>=0:
        #     str_before=str_list[str_index-1]+'。'
        #     long_str=str_before+long_str
        # if str_index+1<str_nums:
        #     str_behind=copy.deepcopy(str_list[str_index+1])
        #     long_str+=('。'+str_behind)
        #分词
        cutting_word_list=list(tokenizer.tokenize(str))
        for i in range(len(cutting_word_list)):
            word=cutting_word_list[i]
            #字在句子中的位置
            #获取标注该字的句子
            an_sentences=copy.deepcopy(cutting_word_list)
            an_sentences.insert(i,'【')
            an_sentences.insert(i+2,'】')
            an_sentences=''.join(an_sentences)
            #将句子换为模型输入的形式
            
            #查找该word是否存在于字典中
            if len(word)==1:
                times=word_dict.count(word)
            else:
                times=0
            if times==1:
                #只有一个释义，输出即可
                words.append(word)
                sentences.append(an_sentences)
                meanings.append(df_dict.iloc[word_dict.index(word)]['words_meaning'])
                second_meanings.append(df_dict.iloc[word_dict.index(word)]['words_meaning'])
            elif times>1:
                start_site=word_dict.index(word)
                word_site=len(''.join(cutting_word_list[0:i]))+len(str_before)
                patch0=list(long_str)
                patch0.insert(word_site,'“')
                patch0.insert(word_site+2,'”')
                patch0=''.join(patch0)
                meanings_index_list=[]
                #计算目标句子和字典句子中对应字的相似性
                sims=[]
                #随机获取字典中不同释义对应的句子
                for j in range(times):
                    temp_m=df_dict.iloc[start_site+j]['words_meaning']
                    patch1=word+'：'+temp_m
                    patch=[patch0,patch1]
                    outputs=model_out(patch)[0][1].item()
                    sims.append(outputs)
                #获取top2的预测释义
                sims=torch.tensor(sims)
                if len(sims)>=2:
                    _, indices = sims.topk(2, dim=0, largest=True, sorted=True)
                    best_choice=indices[0].item()
                    second_choice=indices[1].item()
                else:
                    best_choice=torch.argmax(sims).item()
                    second_choice=torch.argmax(sims).item()
                second_meanings.append(df_dict.iloc[start_site+second_choice]['words_meaning'])
                meanings.append(df_dict.iloc[start_site+best_choice]['words_meaning'])
                sentences.append(an_sentences)
                words.append(word)
        # word_list.append(words)
        word_list+=words
        # word_meaning_list.append(meanings)
        word_meaning_list+=meanings
        second_meaning_list+=second_meanings
        # sentence_list.append(sentences)
        sentence_list+=sentences
    return word_list,word_meaning_list,second_meaning_list,sentence_list

    # for i in range(len(str_list)):
    #     print("------------------------------")
    #     print(str_list[i])
    #     if len(word_list[i])>0:
    #         for j in range(len(word_list[i])):
    #             print("{}: {}".format(word_list[i][j],word_meaning_list[i][j]))

#原句
sentences=[]
#待翻译字
words=[]
#对应意思
meanings=[]
second_meanings=[]
file_names=['大学']
file_path='/root/code/yaoyao/mb/guwen_data/sishu'
obj_path='/root/code/yaoyao/mb/guwen_zhushi'
for file_name in file_names:
    file_name_in="/"+file_name+'.txt'
    file_name_out='/'+file_name+'_contrastive_all_aug_075'+'.txt'
    with open(file_path+file_name_in,'r') as f:
        passage=f.readlines()
    f.close()
    length=len(passage)
    #get noted information in to csv format
    for line in passage:
        i=passage.index(line)
        line=re.sub('\u3000','',line)
        line=re.sub('\n','',line)
        str_list=passage_seg(line)
        w_list,meaning_list,second_meaning_list,s_list=get_noted(str_list)
        if len(w_list)==len(meaning_list) and len(w_list)==len(s_list):
            words+=w_list
            meanings+=meaning_list
            second_meanings+=second_meaning_list
            sentences+=s_list
        else:
            print(w_list)
            print(s_list)
            print(meaning_list)
        print('{}/{}'.format(i,length))
#create dataframe
print('{}/{}/{}'.format(len(words),len(sentences),len(meanings)))
df_dict=pd.DataFrame({'sentence':sentences,'words':words,'words_meaning':meanings,'second_choice':second_meanings})
df_dict.to_csv(obj_path+'/guwen_大学_glossbert4.csv',index=False)



    #降注释后的信息输入到文本文件中
    # with open(obj_path+file_name_out,'a') as f:
    #     f.write("--------------------------------原文-----------------------------")
    #     for line in passage:
    #         line.replace('\u3000','')
    #         line.replace('\n','')
    #         if len(line)>0:
    #             f.write(line)
    #     f.write("--------------------------------注释-----------------------------")
    # #自动注释并输入txt文件
    #     for line in passage:
    #         if len(line)>5 and line.count(' ')<5:
    #             line=re.sub('\u3000','',line)
    #             line=re.sub('\n','',line)
    #             str_list=passage_seg(line)
    #             word_list,word_meaning_list=get_noted(str_list)
                
    #             for i in range(len(str_list)):
    #                 if len(word_list[i])>0:
    #                     f.write("\r\n")
    #                     f.write(str_list[i])
    #                     for j in range(len(word_list[i])):
    #                         f.write('\r\n')
    #                         f.write("{}：{}".format(word_list[i][j], word_meaning_list[i][j]))
    #                     f.write("\r\n")
    # f.close()
    # print(file_name)

