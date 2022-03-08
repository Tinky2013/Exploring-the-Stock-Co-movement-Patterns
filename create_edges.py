import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

K = 12

# fully-connected graph
stock_code = train['thscode'].unique()  # list with the thscode
num_stock = len(stock_code)
# edge_index = []
# for i in range(num_stock):
#     for j in range(i+1, num_stock):
#         edge_index.append([i,j])
#         edge_index.append([j,i])
# edge_index = np.array(edge_index)
# np.save('preprocess_data/edges/adj_index.npy', edge_index)

train_stock, val_stock, test_stock = [], [], []
for i in range(num_stock):
    temp = train[train['thscode'] == stock_code[i]]
    train_stock.append(temp)
    temp = val[val['thscode'] == stock_code[i]]
    val_stock.append(temp)
    temp = test[test['thscode'] == stock_code[i]]
    test_stock.append(temp)

def cal_weight(dt, dtr):
    for t in range(len(dt)-K+1):
        edge_weight = []
        print("processing the day ", t)
        for i in range(num_stock):
            for j in range(i + 1, num_stock):
                K_step_info_i = train_stock[i]['daily_inc'].reset_index(drop=True)[t:t+K]
                K_step_info_j = train_stock[j]['daily_inc'].reset_index(drop=True)[t:t+K]
                dist = K_step_info_i.corr(K_step_info_j, method='pearson')
                edge_weight.append(dist)
                edge_weight.append(dist)

        edge_weight = np.array(edge_weight)
        edge_weight = np.nan_to_num(edge_weight)
        np.save('preprocess_data/edges/weight_'+dtr+'_12K_'+str(t)+'.npy', edge_weight)
        del edge_weight

cal_weight(train, 'train')
cal_weight(val, 'val')
cal_weight(test, 'test')