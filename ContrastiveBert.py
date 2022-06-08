import torch
import torch.nn as nn
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel

class Config(object):
    #配置参数
    def __init__(self,dataset):
        self.model_name='ContrastiveBert_guwen_6sentences'
        #训练集
        self.train_path=dataset+'data/data_con/train_6sentences.csv'
        #测试集
        self.test_path=dataset+'data/data_con/test_6sentences.csv'
        #校验集
        self.dev_path=dataset+'data/data_con/dev_6sentences.csv'
        #模型训练存储
        self.save_path=dataset+'saved_dict_contrastive/'+self.model_name+'trainaug_no_w_add_tags2'+'.ckpt'
        #设备配置
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #超过1000batch还没有提升，提前结束
        self.require_improvement = 1000
        #类别数，2
        #self.num_class = len(self.class_list)
        #epoch数
        self.num_epochs = 3
        self.batch_size = 8
        #句子的向量长度8*11
        self.pad_size = 200
        self.learning_rate=1e-5

        #bert预训练模型位置
        self.bert_path='Ethan-yt'
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        #初始化权重矩阵，用于query和key相乘时用到
        # self.x=init_w(self.hidden_size,0).to(self.device)
        # self.y=init_w(self.hidden_size,0).to(self.device)
        #进行contrastive learning（cross-entropy时的类别数的1/2）
        self.num_classes=self.batch_size
        

#自定义网络中的特征提取层
# class FeatureTransferLayer(nn.Module):
#     def __init__(self,config):
#         super(FeatureTransferLayer,self).__init__()
#         self.device=config.device
#         self.x_weight=torch.nn.Parameter(torch.randn(config.hidden_size,1).to(self.device))
#         self.y_weight=torch.nn.Parameter(torch.randn(1,config.hidden_size).to(self.device))
#         nn.init.xavier_uniform_(self.x_weight)
#         nn.init.xavier_uniform_(self.y_weight)
#     #特征提取层，将anchor对应的向量全部成上768*768的矩阵进行转换
#     def forward(self,anchor_vecs):
#         weight=torch.mm(self.x_weight,self.y_weight)
#         anchor_vecs_w=torch.mm(anchor_vecs,weight).unsqueeze(1)
#         return anchor_vecs_w  

#用于得到整个句子的encoded层的向量
class BertLayer(nn.Module):
    def __init__(self,config):
        super(BertLayer,self).__init__()
        self.bert=AutoModel.from_pretrained(config.bert_path)
        self.device=config.device
        self.batch_size=config.batch_size
        self.hidden_size=config.hidden_size
        for param in self.bert.parameters():
            #直接输出便是False，做微调便用True
            param.requires_grad=True
    
    #获取每个合成句子的bert输出
    #word_sites=[batch_size,类别数（训练时固定为8）]  str_encode=[token_ids,seq_len,masked]
    def forward(self,word_sites,str_encoded):
        #t是词在句子中的出现次数，在训练时默认是0
        #word, sentence是原句子和原古文词，serntence_encode是用于输入句子的内容
        context=str_encoded[0]#输入句子[128,100]
        mask=str_encoded[2]#padding部分进行mask[128,200]
        # tags_token=str_encoded[3]#原句子的token_ids[128,200]
        try:
            ans=self.bert(context,attention_mask=mask,output_hidden_states=False)#[batch_size,100,768]
            ans=ans.last_hidden_state#[batch_size,100,768]
        except IndexError as exception:
            print(word_sites)
            print('---------------')
        #获取对应词在句中的向量,得到分类的值
        key_vecs_batches=[]#[batch_size,7(class_num-1),768]
        anchor_vecs=[]#[batch_size,768]
        for i in range(len(str_encoded[0])):
            if len(str_encoded[0])==1:
                an=ans[0]
                word_site_list=word_sites
            else:
                an=ans[i]
                word_site_list=word_sites[i]
            key_vecs=[]
            for j in range(len(word_site_list)):
                word_site=word_site_list[j]
                if j==0:
                    anchor_vecs.append(an[word_site])
                else:
                    key_vecs.append(an[word_site])
            #将word_vecs转换成tensor
            key_vecs=torch.stack(key_vecs)
            key_vecs_batches.append(key_vecs)
        #输出anchor和key对应的向量
        key_vecs_batches=torch.stack(key_vecs_batches)
        anchor_vecs=torch.stack(anchor_vecs)
        return anchor_vecs,key_vecs_batches
        # #[batch_size,7,768]
        # try:
        #     key_vecs_batches=torch.stack(key_vecs_batches).permute(0,2,1)
        # except RuntimeError as e:
        #     print(word_sites)
        # w=torch.mm(self.x_weight.t(),self.y_weight)
        # anchor_vecs=torch.stack(anchor_vecs)
        # anchor_vecs_w=torch.mm(anchor_vecs,w).unsqueeze(1)
        # #[batch_size,7]
        # classify_val=torch.bmm(anchor_vecs_w,key_vecs_batches)
        # return torch.squeeze(classify_val,1)

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.device=config.device
        self.batch_size=config.batch_size
        self.hidden_size=config.hidden_size
        self.bert=BertLayer(config)
        # self.feature_transfer=FeatureTransferLayer(config)
    def forward(self,word_sites,str_encoded):
        anchor_vecs,key_vecs_batches=self.bert(word_sites,str_encoded)
        # anchor_vecs_w=self.feature_transfer(anchor_vecs)
        classify_val=torch.cosine_similarity(anchor_vecs.unsqueeze(1),key_vecs_batches, dim=2, eps=1e-08)
        # classify_val=torch.bmm(anchor_vecs,key_vecs_batches)
        #[batch_size,7]
        return classify_val
