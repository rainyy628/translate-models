import time
import torch
import numpy as np
from importlib import import_module
#导入模型
import ContrastiveBert as myModel
import contrastive_train
import contrastive_utils

print("here")
if __name__=='__main__':
    dataset='./'#数据集的地址
    #加载模型
    config = myModel.Config(dataset)

    start_time=time.time()
    print("Loading data...")
    train_data,dev_data,test_data=contrastive_utils.build_dataset(config)
    train_iter=contrastive_utils.build_iterator(train_data,config)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic=True#保证每次运行结果相同
    # #enumerate可以同时列出数据和数据下标
    dev_iter=contrastive_utils.build_iterator(dev_data,config)
    test_iter=contrastive_utils.build_iterator(test_data,config)
    #print(dev_iter)

    time_dif=contrastive_utils.get_time_dif(start_time)
    print("Time for preparation:{}".format(time_dif))

    #模型训练，评估与测试
    model=myModel.Model(config).to(config.device)
    contrastive_train.train(config,model,train_iter,dev_iter,test_iter)
    


    
