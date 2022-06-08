#其中放工具
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pandas as pd
import re
import random

#处理成7个孪生网络的输入
PAD, CLS,SEP='[PAD]','[CLS]','[SEP]'

#将数据集中的str分成sentence
def get_s_split(str):
    str=re.sub('[\[,\],\']','',str)
    strs=str.split()
    return strs

#将数据集中的str分成[int,int...]
def get_num_split(str):
    str=re.sub('[\[,\],\']','',str)
    nums=str.split()
    ans=[]
    for num in nums:
        num=int(num)
        ans.append(num)

    return ans


#将句子编码得到bert的输入
def get_encoded(composed_s,config):
    all_infos=[]
    for s in composed_s:
        tokens=[CLS]+config.tokenizer.tokenize(s)
        seq_len=len(tokens)
        tokens_id=config.tokenizer.convert_tokens_to_ids(tokens)
        pad_size=config.pad_size
        if pad_size:
            if len(tokens_id) < pad_size:
                mask = [1] * len(tokens_id) + [0] * (pad_size - len(tokens_id))
                tokens_id = tokens_id + ([0]*(pad_size-len(tokens_id)))
            else:
                mask=[1] * pad_size
                tokens_id= tokens_id[0:pad_size]
                seq_len=pad_size
        all_infos.append((tokens_id,seq_len,mask))
    return all_infos

#word_sites,contents
def load_dataset(file_path,config):
    dataset=pd.read_csv(file_path)
    contents=[]
    data_num=len(dataset)
    print("len_dataset:{}".format(data_num))
    for i in range(data_num):
        word_sites=get_num_split(dataset.iloc[i]['word_sites'])
        if len(word_sites)==0:
            print(len(word_sites))
        composed_s=get_s_split(dataset.iloc[i]['contents'])
        label=int(dataset.iloc[i]['labels'])
        #编译输入格式
        all_infos=get_encoded(composed_s,config)
        contents.append((word_sites,all_infos,label))
        #all_infos [8,3]
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
        #word_sites
        word_sites=[item[0] for item in dataset]
        #encoed_s
        encoded_s=[item[1] for item in dataset] 
        #labels
        labels=[item[2] for item in dataset]
        #token_ids
        sentence_encode=[]
        for i in range(8):
            token_ids=torch.LongTensor([item[i][0] for item in encoded_s]).to(self.device)
            seq_len=torch.LongTensor([item[i][1] for item in encoded_s]).to(self.device)
            mask=torch.LongTensor([item[i][2] for item in encoded_s]).to(self.device)
            sentence_encode.append((token_ids,seq_len,mask))
            #[8_class,3,8]


        return word_sites,sentence_encode,labels
    
    def __next__(self):
        if self.residue and self.index==self.n_batches:
            #最后一批次把所有没有训练过的数据全部输入
            batches=self.dataset[self.index*self.batch_size:len(self.dataset)]
            self.index+=1
            batches=self._to_tensor(batches)
            return batches
        #迭代结束
        elif self.index>=self.n_batches:
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



