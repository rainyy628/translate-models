#用拼接所有的方式做测试(获取embedding时加上词性)
from jiayan import load_lm
from jiayan import CRFSentencizer,CharHMMTokenizer
import ContrastiveBert as myModel
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import re
import random
import copy
import udkanbun 

lzh=udkanbun.load()

PAD, CLS,SEP='[PAD]','[CLS]','[SEP]'
tags_list=['[PAD]','VERB', 'NOUN', 'PUNCT', 'ADV', 'CCONJ', 'PRON', 'SCONJ', 'INTJ', 'PART', 'AUX', 'NUM', 'PROPN', 'SYM', 'ADP', 'X','[CLS]','[SEP]']
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")

model = AutoModel.from_pretrained("ethanyt/guwenbert-base")

#读取字典
df_dict=pd.read_csv('/root/code/yaoyao/mb/data_origin/yao_dict_xc_amend.csv')
#获取字列表
word_dict=df_dict['words'].tolist()

lm=load_lm('jiayan.klm')
sentencizer=CRFSentencizer(lm)
sentencizer.load('cut_model')

#分词
# text="停车做爱枫林晚"
tokenizer=CharHMMTokenizer(lm)
# print(list(tokenizer.tokenize(text)))

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
model_name='ContrastiveBert'
config=myModel.Config(project_dict)

conbert_model=myModel.Model(config).to(config.device)
conbert_model.load_state_dict(torch.load(config.save_path))


conbert_model.eval()



#每次输出结果相同
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(4)
# torch.backends.cudnn.deterministic=True

def model_out(word_sites,composed_s):
    s_encoded=get_encoded(composed_s)
    tokens_id=s_encoded[0]
    seq_len=s_encoded[1]
    mask=s_encoded[2]
    pos_tags=s_encoded[3]
    tokens_id=torch.LongTensor([tokens_id]).to(torch.device('cuda'))
    seq_len=torch.LongTensor([seq_len]).to(torch.device('cuda'))
    mask=torch.LongTensor([mask]).to(torch.device('cuda'))
    pos_tags=torch.LongTensor([pos_tags]).to(torch.device('cuda'))
    s_encoded=[tokens_id,seq_len,mask,pos_tags]
    with torch.no_grad():
        output=conbert_model(word_sites,s_encoded)
    return output

def get_pos_tag(s):
    tags=[]
    s_tags=lzh(s)
    #num记录总共转换成词性的个数
    num=0
    for i in range(len(s_tags)):
        try:
            tag=s_tags[i+1].upos
            #有的多个字对应一个tag
            lemma=s_tags[i+1].lemma
            lemma_len=len(lemma)
        except IndexError as exception:
            print('s:{} length:{} tags_num:{}'.format(s,len(s),len(s_tags)))
            print(i+1)
            print('---------------------')
        for j in range(lemma_len):
            tags.append(tag)
            num+=1
        if num>=len(s):
            break
    return tags

#词性数据取tokenize
def tags_tokenize(pos_tags):
    tags_tokenize=[]
    for tags in pos_tags:
        tags_tokenize.append(tags_list.index(tags))
    return tags_tokenize

def get_encoded(composed_s):
    tokens=[CLS]
    pos_tags_tokens=[CLS]
    for s in composed_s:
        token=config.tokenizer.tokenize(s)+[SEP]
        tokens+=token
        #转换成词性
        pos_tags_token=get_pos_tag(s)
        pos_tags_tokens+=(pos_tags_token+[SEP])
    #去掉最后一个SEP
    tokens.pop()
    pos_tags_tokens.pop()
    seq_len=len(tokens)
    tokens_id=config.tokenizer.convert_tokens_to_ids(tokens)
    tags_ids=tags_tokenize(pos_tags_tokens)
    mask = [1] * len(tokens_id)
    return tokens_id,seq_len,mask,tags_ids

#获取待翻译字前后各10个字,以及现在他在句子中的位置
def get_sentence_cut(w_site,s) :
    new_s=[]
    new_index=0
    new_s.append(s[w_site])
    for i in range(1,11):
        if w_site-i>=0:
            new_s.insert(0,s[w_site-i])
            new_index+=1
        if w_site+i<len(s):
            new_s.append(s[w_site+i])
    return new_index,''.join(new_s)

def get_s_split(str):
    str=re.sub('[\[,\],\']','',str)
    strs=str.split()
    return strs

#将一段文章进行自动注释
def get_noted(str_list):
    word_list=[]
    word_meaning_list=[]
    second_words_meaning_list=[]
    sentence_list=[]
    str_nums=len(str_list)
    for str_index in range(str_nums):
        str=str_list[str_index]
        str=re.sub('\u3000','',str)
        words=[]
        meanings=[]
        #第二选择
        second_meaning=[]
        sentences=[]
        #获取更长长度的句子
        long_str=copy.deepcopy(str)
        str_before=''
        str_behind=''
        if str_index-1>=0:
            str_before=str_list[str_index-1]+'。'
            long_str=str_before+long_str
        if str_index+1<str_nums:
            str_behind=copy.deepcopy(str_list[str_index+1])
            long_str+=('。'+str_behind)
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
            word_sites_list=[]
            s_composed_list=[]
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
                second_meaning.append(df_dict.iloc[word_dict.index(word)]['words_meaning'])
            elif times>1:
                start_site=word_dict.index(word)
                #获取该字在本句子中前10和后10的字符
                #获取该字在本句子中的index【加上[CLS]】
                word_site=len(''.join(cutting_word_list[0:i]))+len(str_before)
                sites_new,str_new=get_sentence_cut(word_site,long_str)
                word_sites_list.append(sites_new)
                s_composed_list.append(str_new)
                word_sites_origin=copy.deepcopy(word_sites_list)
                s_composed_origin=copy.deepcopy(s_composed_list)
                meanings_index_list=[]
                #计算目标句子和字典句子中对应字的相似性
                #随机获取字典中不同释义对应的句子
                #由于每次例子中取到的是不同的，所以用循环多次获取
                for _t in range(10):
                    for j in range(times):
                        temp_s=df_dict.iloc[start_site+j]['sentences']
                        if len(temp_s)>0:
                            examples_list=get_s_split(temp_s)
                            if len(examples_list)>0:
                                example_choose=random.choice(examples_list)
                                temp_word_site=example_choose.index(word)
                                sites_new,example_new=get_sentence_cut(temp_word_site,example_choose)
                                word_sites_list.append(sites_new)
                                s_composed_list.append(example_new)

                                #将所有释义的例子都进行拼接
                                # for example in examples_list:
                                #     s_composed_list.append(example)
                                #     word_sites_list.append(example.index(word)+len(long_str)+2)#加上[CLS]以及[SEP]
                                #     #改变word_sites
                                #     try:
                                #         outputs=model_out(word_sites_list,s_composed_list)[0].item()
                                #     except IndexError as exception:
                                #         print(word_sites_list)
                                #         print(s_composed_list)
                                #         print(long_str)
                                #         print(str_before)
                                #     word_sim+=outputs
                                #     s_composed_list.pop()
                                #     word_sites_list.pop()
                                # sims.append(word_sim/len(examples_list))
                                #仅在第一次循环时计算
                                if _t==0:
                                    meanings_index_list.append(j)
                    #改变word_sites
                    for k in range(len(word_sites_list)):
                        word_sites_list[k]=word_sites_list[k]+(1+k)+len(''.join(s_composed_list[0:k]))
                    #得到模型输出
                    outputs=model_out(word_sites_list,s_composed_list)
                    word_sites_list=copy.deepcopy(word_sites_origin)
                    s_composed_list=copy.deepcopy(s_composed_origin)
                    if _t==0:
                        out_all=outputs
                    else:
                        out_all+=outputs
                outputs=out_all/10
                outputs=outputs[0]
                #最合适的意思的下标
                # best_choice=torch.argmax(outputs).item()
                if len(outputs)>=2:
                    _, indices = outputs.topk(2, dim=0, largest=True, sorted=True)
                    best_choice=indices[0].item()
                    second_choice=indices[1].item()
                else:
                    best_choice=torch.argmax(outputs).item()
                    second_choice=torch.argmax(outputs).item()
                meanings.append(df_dict.iloc[start_site+meanings_index_list[best_choice]]['words_meaning'])
                second_meaning.append(df_dict.iloc[start_site+meanings_index_list[second_choice]]['words_meaning'])
                sentences.append(an_sentences)
                words.append(word)
        # word_list.append(words)
        word_list+=words
        # word_meaning_list.append(meanings)
        word_meaning_list+=meanings
        # sentence_list.append(sentences)
        sentence_list+=sentences
        second_words_meaning_list+=second_meaning
    return word_list,word_meaning_list,second_words_meaning_list,sentence_list

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
#第二选择
second_meaning=[]
file_names=['大学']
file_path='/root/code/yaoyao/mb/guwen_data/sishu'
obj_path='/root/code/yaoyao/mb/guwen_zhushi'
for file_name in file_names:
    file_name_in="/"+file_name+'.txt'
    file_name_out='/'+file_name+'_contrastive_sb_all_aug_075'+'.txt'
    with open(file_path+file_name_in,'r') as f:
        passage=f.readlines()
    length=len(passage)
    f.close()
    #get noted information in to csv format
    for line in passage:
        i=passage.index(line)
        line=re.sub('\u3000','',line)
        line=re.sub('\n','',line)
        str_list=passage_seg(line)
        w_list,meaning_list,second_meaning_list,s_list=get_noted(str_list)
        if len(w_list)==len(meaning_list) and len(w_list)==len(s_list) and len(w_list)==len(second_meaning_list):
            words+=w_list
            meanings+=meaning_list
            second_meaning+=second_meaning_list
            sentences+=s_list
        else:
            print(w_list)
            print(s_list)
            print(meaning_list)
            print(second_meaning_list)
        if i%10==0:
            print('{}/{}'.format(i,length))
#create dataframe
print('{}/{}/{}'.format(len(words),len(sentences),len(meanings)))
df_dict=pd.DataFrame({'sentence':sentences,'words':words,'words_meaning':meanings,'second_choice':second_meaning})
df_dict.to_csv(obj_path+'/guwen_大学_con_add_postags.csv',index=False)



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

