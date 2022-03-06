import numpy as np
import torch
import  torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
from torch.nn import TripletMarginLoss
import utils

def train(config,model,train_iter,dev_iter,test_iter):
    start_time=time.time()
    model.train()

    param_optimizer=list(model.named_parameters())
    #不需要衰减的参数
    no_decay=['bias','LayerNorm','LayerNorm.weight']

    #参数是否需要衰减
    #n=name p=param
    optimizer_grouped_parameters=[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight"_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]

    optimizer=BertAdam(optimizer_grouped_parameters,
                       lr=config.learning_rate,
                       warmup=0.05,
                       t_total=len(train_iter)*config.num_epochs)
    #记录进行batch数
    total_batch=0
    #记录校验集合最好的loss
    dev_best_loss=float('inf')
    #记录上次校验集loss下降的batch数
    last_improve=0
    #记录是否很久没有效果提升，是否停止训练
    flag=False
    #loss
    triplet_loss=TripletMarginLoss(margin=1.0,p=2)
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch+1,config.num_epochs))
        for i,(trains) in enumerate(train_iter):
            #print(trains[0])
            torch.cuda.empty_cache()
            outputs=model(trains[0],trains[1],trains[2],trains[3],trains[4],trains[5],trains[6],trains[7],trains[8])
            # print("len_outpus:{}".format(type(outputs[0][0])))
            torch.cuda.empty_cache()
            model.zero_grad()
            #outputs[:][:][]：tensor
            # anchor=torch.LongTensor(outputs[0])
            # positive=torch.LongTensor(outputs[1])
            # negtive=torch.LongTensor(outputs[2])
            for j in range(len(outputs[0])):
                if j==0:
                    loss=triplet_loss(outputs[0][j],outputs[1][j],outputs[2][j])
                else:
                    loss+=triplet_loss(outputs[0][j],outputs[1][j],outputs[2][j])
            loss=loss/float(len(outputs[0]))
            print("loss:{}".format(loss))
            loss.backward()
            # print("----loss-----")
            optimizer.step()
            #每一百次输出一次训练日志
            if total_batch%100==0:
                # losses=loss.data.cpu()
                dev_loss=evaluate(config,model,dev_iter)
                #loss降低
                if dev_loss < dev_best_loss:
                    dev_best_loss=dev_loss
                    torch.save(model.state_dict(),config.save_path)
                    improve="*"
                    last_improve=total_batch
                #loss没有降低
                else:
                    improve="-"
                time_dif=utils.get_time_dif(start_time)
                msg='Iter:{:>6}, Train Loss:{:>5.2f},Val Loss:{:>5.2f}, Time:{} {}'
                print(msg.format(total_batch,loss.item(),dev_loss,time_dif,improve))
                #之前调用过eval
                model.train()
            total_batch=total_batch+1
            #最大限次没有参数提升，停止训练
            if total_batch-last_improve>config.require_improvement:
                print("Long Time without optimization. Training Termimated!")
                flag=True
                break
        if flag:
            break
    test(config, model, test_iter)


#计算验证集的loss
def evaluate(config,model,dev_iter):
    model.eval()
    loss_total=0
    triplet_loss = TripletMarginLoss(margin=1.0, p=2)
    with torch.no_grad():
        for inputs in dev_iter:
            outputs=model(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],inputs[5],inputs[6],inputs[7],inputs[8])
            for j in range(len(outputs[0])):
                if j==0:
                    loss=triplet_loss(outputs[0][j],outputs[1][j],outputs[2][j])
                else:
                    loss+=triplet_loss(outputs[0][j],outputs[1][j],outputs[2][j])
            loss=loss/float(len(outputs[0]))
            loss_total=loss_total+loss
    return loss_total/len(dev_iter)

def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time=time.time()
    test_loss=evaluate(config,model,test_iter)
    time_dif=utils.get_time_dif(start_time)
    msg='Test Loss:{:>5.2f}, Time Cost:{}'
    print(msg.format(test_loss,time_dif))
