#其中放工具
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pandas as pd
import re

PAD, CLS='[PAD]','[CLS]'

#将数据集中的str分成word 和sentence
def get_split(str):
    str=re.sub('[\[,\],\']','',str)
    strs=str.split()
    # print("----")
    # print(strs)
    word=strs[0]
    sentence=strs[1]
    # print("word:{} sentence:{}".format(word,sentence))
    return word,sentence
    

#将句子编码得到bert的输入
def get_encoded(s,config):
    token=config.tokenizer.tokenize(s)
    token=[CLS]+token
    seq_len=len(token)
    token_ids=config.tokenizer.convert_tokens_to_ids(token)
    pad_size=config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token) + [0] * (pad_size - len(token))
            token_ids = token_ids + ([0]*(pad_size-len(token)))
        else:
            mask=[1] * pad_size
            token_ids= token_ids[:pad_size]
            seq_len=pad_size
    return token_ids,seq_len,mask

def load_dataset(file_path,config):
    #每次输入有三组，分别是anchor，positive，negtive
    #每组里面包含word，sentece以及bert的输入格式
    #输出内容
    #原词
    dataset=pd.read_csv(file_path)
    item='PythonTab'
    contents=[]

    data_num=len(dataset)
    print("len_dataset:{}".format(data_num))
    for site in range(data_num):
        word1,sentence1=get_split(dataset.loc[site]['anchor'])
        word2,sentence2=get_split(dataset.loc[site]['positive'])
        word3,sentence3=get_split(dataset.loc[site]['negtive'])
        #编译输入格式
        token_id1,seq_len1,mask1=get_encoded(sentence1,config)
        token_id2,seq_len2,mask2=get_encoded(sentence2,config)
        token_id3,seq_len3,mask3=get_encoded(sentence3,config)
        contents.append((word1,sentence1,(token_id1,seq_len1,mask1),word2,sentence2,(token_id2,seq_len2,mask2),word3,sentence3,(token_id3,seq_len3,mask3)))
    print(len(contents))
    return contents



def build_dataset(config):
    #return train, dev, test
    train = load_dataset(config.train_path,config)
    dev = load_dataset(config.dev_path,config)
    test = load_dataset(config.test_path,config)
    return train, dev, test

class DatasetIterator(object):
    def __init__(self,dataset,batch_size,device):
        self.dataset=dataset
        self.batch_size=batch_size
        self.device=device
        #//是整数除法
        #index记录批次号
        self.index=0
        self.n_batches=len(dataset)//batch_size
        # print("n_batches={}".format(self.n_batches))
        self.residue=False#batch数量是整数
        if len(dataset) % self.n_batches!=0:
            self.residue=True

    #将数据转换成容易直接输入bert的形式
    def _to_tensor(self,dataset):
        #datas[word1,sentence1,sentence1_encode,word2,sentence2,sentence2_encode,word3,sentence3,sentence3_encode]
        #word
        # print("len of dataset2:{}".format(len(dataset[2])))
        word1=[item[0] for item in dataset]
        word2=[item[3] for item in dataset]
        word3=[item[6] for item in dataset]
        #sentencen
        sentence1=[item[1] for item in dataset]
        sentence2=[item[4] for item in dataset]
        sentence3=[item[7] for item in dataset]
        #sentence_encode
        sentence1_encode=[item[2] for item in dataset]
        sentence2_encode=[item[5] for item in dataset]
        sentence3_encode=[item[8] for item in dataset]
        # print(sentence1_encode[3])
        #token_id
        token_ids1=torch.LongTensor([item[0] for item in sentence1_encode]).to(self.device)
        token_ids2 = torch.LongTensor([item[0] for item in sentence2_encode]).to(self.device)
        token_ids3 = torch.LongTensor([item[0] for item in sentence3_encode]).to(self.device)
        # seq_len
        seq_len1=torch.LongTensor([item[1] for item in sentence1_encode]).to(self.device)
        seq_len2 = torch.LongTensor([item[1] for item in sentence2_encode]).to(self.device)
        seq_len3 = torch.LongTensor([item[1] for item in sentence3_encode]).to(self.device)
        #mask
        mask1=torch.LongTensor([item[2] for item in sentence1_encode]).to(self.device)
        mask2 = torch.LongTensor([item[2] for item in sentence2_encode]).to(self.device)
        mask3 = torch.LongTensor([item[2] for item in sentence3_encode]).to(self.device)

        sentence1_encode=[token_ids1,seq_len1,mask1]
        sentence2_encode=[token_ids2,seq_len2,mask2]
        sentence3_encode=[token_ids3,seq_len3,mask3]

        return word1,sentence1,sentence1_encode,word2,sentence2,sentence2_encode,word3,sentence3,sentence3_encode

    def __next__(self):
        if self.residue and self.index==self.n_batches:
            #最后一批次把所有没有训练过的数据全部输入
            batches=self.dataset[self.index*self.batch_size:len(self.dataset)]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
        #迭代结束
        elif self.index>self.n_batches:
            self.index=0
            raise StopIteration
        #普通情况
        else:
            batches=self.dataset[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index+=1
            # print("-----")
            # print(len(batches))
            batches=self._to_tensor(batches)
            return batches


        
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            #当n_batches不是整数时
            return self.n_batches+1
        else:
            return self.n_batches


def build_iterator(dataset,config):
    iter = DatasetIterator(dataset,config.batch_size,config.device)
    return iter

#经过了多少时间
def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))
