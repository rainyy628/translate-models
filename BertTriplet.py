import torch
import torch.nn as nn
import re
import numpy as np
from pytorch_pretrained import BertTokenizer,BertModel

class Config(object):
    #配置参数
    def __init__(self,dataset):
        self.model_name='BertTriplet'
        #训练集
        self.train_path=dataset+'/data/trainn.csv'
        #测试集
        self.test_path=dataset+'/data/testt.csv'
        #校验集
        self.dev_path=dataset+'/data/devv.csv'
        #类别，去除前后的空格，相似 不相似
        #self.class_list=[x.strip() for x in open(dataset+'/data/calss.txt').readlines()]
        #模型训练存储
        self.save_path=dataset+'saved_dict'+self.model_name+'.ckpt'
        #设备配置
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #超过1000batch还没有提升，提前结束
        self.require_improvement = 200
        #类别数，2
        #self.num_class = len(self.class_list)
        #epoch数
        self.num_epochs = 3
        self.batch_size = 8
        #句子的向量长度
        self.pad_size = 200
        self.learning_rate=1e-5

        #bert预训练模型位置
        self.bert_path='bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

#用于得到整个句子的encoded层的向量
class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_path)
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        for param in self.bert.parameters():
            #直接输出便是False，做微调便用True
            param.requires_grad=True

    #获取每个句子的bert输入
    def forward(self,word1,sentence1,sentence1_encoded,word2,sentence2,sentence2_encoded,word3,sentence3,sentence3_encoded):
        #x sentence分别是anchor,positive,negtive
        #word, sentence是原句子和原古文词，serntence_encode是用于输入句子的内容
        context1=sentence1_encoded[0]#输入句子[128,200]
        mask1=sentence1_encoded[2]#padding部分进行mask[128,200]
        ans1,_=self.bert(context1,attention_mask=mask1,output_all_encoded_layers=True)#[127,200,768]

        context2 = sentence2_encoded[0]  # 输入句子[128,200]
        mask2 = sentence2_encoded[2]  # padding部分进行mask[128,200]
        ans2, _ = self.bert(context2, attention_mask=mask2, output_all_encoded_layers=True)  # [127,200,768]

        context3 = sentence3_encoded[0]  # 输入句子[128,200]
        mask3 = sentence3_encoded[2]  # padding部分进行mask[128,200]
        ans3, _ = self.bert(context3, attention_mask=mask3, output_all_encoded_layers=True)  # [127,200,768]
        #获取对应词在句中的向量,
        word_vec1=GetWordVec(word1,sentence1,ans1[0])
        word_vec2 = GetWordVec(word2, sentence2, ans2[0])
        word_vec3 = GetWordVec(word3, sentence3, ans3[0])

        #使用triplet

        return word_vec1,word_vec2,word_vec3

#用于获取句子对应的目标子向量[8,200,768] [batch_size,padding,..]
#bacht_size:list inside:Tensor
def GetWordVec(words,sentences,vecs):
    #vecs是句子的embedding
    #word,sentence,均是list
    output=[]
    for t in range(len(words)):
        word=words[t]
        sentence=sentences[t]
        vec=vecs[0]
        res = sentence.find(word)
        length = len(word)
        # 获取最终输出值
        # print("vec_shape:{}".format(vec.shape))
        for i in range(0, length):
            if i == 0:
                sum = vec[res+1]
            else:
                # keepdims=true
                sum += vec[res+i+1]
        output.append(sum/length)
    return output


def evaluate(v1,v2):
    #计算余弦相似度
    import numpy as np
    v1=v1/np.linalg.norm(v1)
    v2=v2/np.linalg.norm(v2)
    return v1.dot(v2)




