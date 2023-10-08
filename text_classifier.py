from __future__ import print_function


import gc
import numpy as np
import pandas as pd
import pickle
import warnings
from tqdm import tqdm

from classifier_utils import *
from sklearn.metrics import accuracy_score

def run_model(train_txt, test_txt, num_classes, indices):
    warnings.filterwarnings("ignore")
    train_x, train_y = get_x_y(train_txt, num_classes, 300, input_size, word2vec, indices)

    test_x, test_y = get_x_y(test_txt, num_classes, 300, input_size, word2vec, range(lengths[task_name][1]))
    batch_size=64

    train_data,train_dl=to_dataset(train_x,train_y,batch_size)

    model=RNNModel(num_classes,drop_rate=0.5)
    optimizer=torch.optim.Adam(model.parameters())

    model.cuda()
    model.train()

    ls=np.zeros(10)

    for ep in range(200):
        ep_loss=0
        for idx, (x,y) in enumerate(train_dl):
                
                optimizer.zero_grad()
            
                preds=model(x)
                criterion=CrossEntropyLoss().cuda()  
                loss=criterion(preds,y)						#backpropagate loss
                loss.backward()
                optimizer.step()
                
                x.cpu()
                y.cpu()
                
                ep_loss+=loss;	
        
        ep_loss=ep_loss/(idx+1)
        ls[ep%10]=ep_loss                                   #最近10个epoch的loss

        if (ep+1)%50==0 or ep==0: 									#每50 epoch 打印
            print("The average loss at ",ep+1," epoch is",ep_loss.item())

        if (ep+1)%10==0:
            if(np.var(ls)<0.00001):									#early stop
                print("Early Stopped at epoch: ",ep+1,"\n")
                break

    model.eval()
    with torch.no_grad():	            #在测试集上检测
            
            x=torch.tensor(test_x).float().cuda()
            preds=model(x)
            preds=one_hot_to_categorical(preds.cpu().numpy())
            test_y=one_hot_to_categorical(test_y)
            acc=accuracy_score(test_y, preds)
            return acc



if __name__ == "__main__":


    dataset={'stsa2':[2,30], 'TREC':[6,22]}
    increments=[0.01,0.05,0.1,0.2,0.6,1]
    lengths={'stsa2':[6228,1821],"TREC":[4906,500]}

    task_name='stsa2'
    num_classes=dataset[task_name][0]
    input_size=dataset[task_name][1]

    train_orig = 'datasets/'+task_name+'/train.tsv'
    train_aug="aug_data/"+task_name+"/train.tsv"
    test_path="aug_data/"+task_name+"/test.tsv"
    word2vec=pickle.load(open("aug_data/"+task_name+"/word2vec_pickle", 'rb'))


    for increment in increments:
		
        l=lengths[task_name][0]
        ind=np.random.choice(l,int(l*increment),replace=False)       #选出下标
        aug_ind=np.hstack((ind,ind+l,ind+2*l) )       #根据ind选出增强数据下标

        #calculate original accuracy
        orig_acc = run_model(train_orig, test_path, num_classes, ind)
		#calculate CBERT augmented accuracy
        aug_acc = run_model(train_aug, test_path, num_classes, aug_ind)


        print(task_name, "with %" ,100*increment, "data has accuracy:\n", 
		orig_acc, " with original data and ", aug_acc, " with CBERT\n\n")
	
        new_data = [increment, orig_acc, aug_acc]

		# 将新数据转换为DataFrame
        new_row = pd.DataFrame([new_data])
		# 将新行添加到现有的CSV文件
        new_row.to_csv('Res/'+task_name+'_accuracy.csv', mode='a', header=False, index=False)
        gc.collect()