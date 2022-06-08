import torch
import torch.nn as nn
import re
import numpy as np
from pytorch_pretrained import BertTokenizer,BertModel

class Config(object):
    #配置参数
    def __init__(self,dataset):
        self.model_name='GlossBert2'
        #训练集
        self.train_path=dataset+'/data/glossbert/train.csv'
        #测试集
        self.test_path=dataset+'/data/glossbert/test.csv'
        #校验集
        self.dev_path=dataset+'/data/glossbert/dev.csv'
        #模型训练存储
        self.save_path=dataset+'/saved_dict_gloss/'+self.model_name+'.ckpt'
        #设备配置
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #超过1000batch还没有提升，提前结束
        self.require_improvement = 1000
        #类别数，2
        #self.num_class = len(self.class_list)
        #epoch数
        self.num_epochs = 4
        self.batch_size = 64
        #句子的向量长度8*11
        self.pad_size = 50
        self.learning_rate=2e-5

        #bert预训练模型位置
        self.bert_path='bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
    
        self.num_classes=self.batch_size
        
#用于得到整个句子的encoded层的向量
class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_path)
        self.device=config.device
        self.batch_size=config.batch_size
        self.hidden_size=config.hidden_size
        for param in self.bert.parameters():
            #直接输出便是False，做微调便用True
            param.requires_grad=True
        self.fc = nn.Linear(config.hidden_size, 2)
        
    def forward(self,str_encoded):
        context = str_encoded[0]  # 输入的句子
        mask = str_encoded[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

