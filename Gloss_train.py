import numpy as np
import torch
import  torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
import contrastive_split_utils


def train(config,model,train_iter,dev_iter,test_iter):
    start_time=time.time()
    model.train()

    param_optimizer=list(model.named_parameters())
    
    #不需要衰减的参数
    no_decay=['bias','LayerNorm','LayerNorm.weight']

    #参数是否需要衰减
    #n=name p=param
    optimizer_grouped_parameters=[
        {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]

    optimizer=BertAdam(optimizer_grouped_parameters,
                       lr=config.learning_rate,
                       warmup=0.05,
                       t_total=len(train_iter)*config.num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)
    #记录进行batch数
    total_batch=0
    #记录校验集合最好的loss
    dev_best_loss=float('inf')
    #记录上次校验集loss下降的batch数
    last_improve=0
    #记录是否很久没有效果提升，是否停止训练
    flag=False
    model.train()
    train_acc=0
    for epoch in range(config.num_epochs):
        print('Epoch{}/{}'.format(epoch+1,config.num_epochs))
        for i,(trains) in enumerate(train_iter):
            torch.cuda.empty_cache()
            labels=torch.tensor(trains[1]).to(config.device)
            outputs=model(trains[0])
            torch.cuda.empty_cache()
            model.zero_grad(torch)
            loss=F.cross_entropy(outputs,labels)
            loss.backward()
                
            optimizer.step()
            scheduler.step()
            #每一百次输出一次训练日志
            if total_batch%100==0:
                #计算准确度
                true = labels.data.cpu()
                outputs.data
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_loss,dev_acc=evaluate(config,model,dev_iter)
                #loss降低
                if dev_loss < dev_best_loss:
                    dev_best_loss=dev_loss
                    model.state_dict()
                    torch.save(model.state_dict(),config.save_path)
                    improve="*"
                    last_improve=total_batch
                #loss没有降低
                else:
                    improve="-"
                time_dif=contrastive_split_utils.get_time_dif(start_time)
                msg='Iter:{:>6}, Train Loss:{:>6.4f},Train acc:{:>6.2%},Val Loss:{:>6.4f}, Dev acc:{:>6.2%},Time:{} {}'
                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc,time_dif,improve))
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


def evaluate(config,model,dev_iter):
    model.eval()
    loss_total=0
    acc_total=0
    acc_d=0
    acc_n=0
    with torch.no_grad():
        for inputs in dev_iter:
            labels=torch.tensor(inputs[1]).to(config.device)
            outputs=model(inputs[0])
            #计算准确度
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            #计算准确度
            for i in range(len(true)):
                if true[i]==1:
                    acc_d+=1
                    if predic[i]==1:
                        acc_n+=1
            torch.cuda.empty_cache()
            loss=F.cross_entropy(outputs,labels)
            loss_total=loss_total+loss
    length=len(dev_iter)
    return loss_total/length,acc_n/acc_d


def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time=time.time()
    test_loss,test_acc=evaluate(config,model,test_iter)
    time_dif=contrastive_split_utils.get_time_dif(start_time)
    msg='Test Loss:{:>5.2f},Test acc:{:>6.2%}, Time Cost:{}'
    print(msg.format(test_loss,test_acc,time_dif))


