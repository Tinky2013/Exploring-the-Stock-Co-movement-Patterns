import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

from sklearn.metrics import mean_squared_error

param = {
    # training parameters
    # 'trainDt_path': 'combine/2018_zz500.csv',
    # 'testDt_path': 'combine/2019_zz500.csv',
    'trainDt_path': 'train.csv',
    'testDt_path': 'test.csv',
    'TRAIN': True,
    'TEST': True,

    'lr': 0.005,
    'save_model_name': 'model_save/train/ts5_all',
    'test_model_name': 'model_save/train/ts5_all_18000',
    'img_path': 'img/ts5_all_18000',
    'epoch': 1000,
    'save_freq': 100,

    'window_size_K': 12,
    'batch_size': 8,
    'future_ts': 5,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataInitializer():
    def __init__(self, param):
        # read the data and get the time stumps
        self.train_dt = pd.read_csv('train.csv')
        self.val_dt = pd.read_csv('val.csv')
        self.test_dt = pd.read_csv('test.csv')

        self.train_ts = self.train_dt[self.train_dt['thscode']=='000008.SZ']['time']
        self.val_ts = self.val_dt[self.val_dt['thscode']=='000008.SZ']['time']
        self.test_ts = self.test_dt[self.test_dt['thscode']=='000008.SZ']['time']

        self.K = param['window_size_K']
        self.batch_size = param['batch_size']
        self.future_ts = param['future_ts']

        # index for processing
        self.train_size, self.val_size, self.test_size = len(self.train_ts), len(self.val_ts), len(self.test_ts)
        if self.train_size <=  self.K + self.batch_size + self.future_ts:
            raise ValueError("Time series too short!")

        self.train_idx = np.arange(self.train_size)
        self.val_idx = np.arange(self.train_size, self.train_size+self.val_size)
        self.test_idx = np.arange(self.train_size+self.val_size, self.train_size+self.val_size+self.test_size)

        # fully-connected graph
        self.stock_code = self.train_dt['thscode'].unique()  # list with the thscode
        self.num_stock = len(self.stock_code)
        self.edge_index = np.load('preprocess_data/edges/adj_index.npy')

    # calculate the edge weight at time t based on the previous K-step history
    def calEdgesWeight(self, dtr, t):
        edge_weight = np.load('preprocess_data/edges/weight_'+dtr+'_12K_'+str(t)+'.npy')
        return edge_weight

    def getNodesLabel(self, dt, t):
        label = []
        for i in range(self.num_stock):
            rank = dt[dt['thscode']==self.stock_code[i]]['return_rank'][t-1,t].reset_index(drop=True)
            label.append(rank[0])
        # label: list (num_stock)
        return label

    def getNodesFeature(self, dt, t, K):
        feature = []
        for i in range(self.num_stock):
            state = dt[dt['thscode']==self.stock_code[i]].iloc[:,3:]
            feature.append(np.array(state[t-1:t])[0])
        # feature: list (num_stock, feature_dim)
        return feature

    # define how to get one sample
    def constructGraph(self, dt, dtr, t, K):
        edge_index = torch.LongTensor(self.edge_index)
        edge_weight = torch.FloatTensor(self.calEdgesWeight(dtr, t))
        node_feature = torch.FloatTensor(self.getNodesFeature(dt, t, K))
        node_labels = torch.LongTensor(self.getNodesLabel(dt, t)) # int ranking label
        return edge_index, edge_weight, node_feature, node_labels

    # create the graph dataset
    def createGraphList(self, K):
        trainG, valG, testG = [], [], []
        for t in range(K, self.train_size):
            edge_index, edge_weight, node_feature, node_labels = self.constructGraph(self.train_dt, 'train', t, K)
            graph_t = Data(x=node_feature, edge_index=edge_index.t().contiguous(), edge_attr=edge_weight, y=node_labels)
            trainG.append(graph_t)
        for t in range(K, self.val_size):
            print(t)
            edge_index, edge_weight, node_feature, node_labels = self.constructGraph(self.val_dt, 'val', t, K)
            graph_t = Data(x=node_feature, edge_index=edge_index.t().contiguous(), edge_attr=edge_weight, y=node_labels)
            trainG.append(graph_t)
        for t in range(K, self.test_size):
            edge_index, edge_weight, node_feature, node_labels = self.constructGraph(self.test_dt, 'test', t, K)
            graph_t = Data(x=node_feature, edge_index=edge_index.t().contiguous(), edge_attr=edge_weight, y=node_labels)
            trainG.append(graph_t)
        return trainG, valG, testG


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        pass

    def forward(self, input):
        print(input)
        print(input.shape)
        return None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def trainModel():
    print("loading")
    t0 = time.time()
    train_loss, minL = [], np.inf
    for epoch in range(param['epoch']):
        model.train()
        for train_data in trainLoader:
            train_data = train_data.to(device)
            output = model(train_data)
            loss = F.mse_loss(output, train_data.y)
            optimizer.zero_grad()
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        if epoch % param['save_freq'] == 0:
            # save every 'save_freq' step
            torch.save(model.state_dict(), param['save_model_name'] + '_' + str(epoch) + '.pkl')
            t1 = time.time()
            print("Train epi:", epoch, " | total time:", t1-t0, " | Ave loss:", np.mean(train_loss[-param['save_freq']:]))

            # model evaluation
            model.eval()
            val_label_list, val_pred_list = [], []
            for val_data in valLoader:
                val_data = val_data.to(device)
                val_label_list.extend(val_data.y.detach().cpu().numpy())
                pred = model(val_data)
                val_pred_list.extend(pred.detach().cpu().numpy())

            val_loss = mean_squared_error(val_label_list, val_pred_list)
            if val_loss <= minL:
                torch.save(model.state_dict(), param['save_model_name']+'_best.pkl')
                print("Saving best reward at iteration ", epoch, "with val loss of:", val_loss)
                minL = val_loss

    torch.save(model.state_dict(), param['save_model_name']+'_final.pkl')

def testModel():
    model.load_state_dict(torch.load(param['test_model_name'] + '.pkl'))
    model.eval()
    train_label_list, train_pred_list = [], []
    for train_data in trainLoader:
        train_data = train_data.to(device)
        train_label_list.extend(train_data.y.detach().cpu().numpy())
        pred = model(train_data)
        train_pred_list.extend(pred.detach().cpu().numpy())
    test_label_list, test_pred_list = [], []
    for test_data in testLoader:
        test_data = test_data.to(device)
        test_label_list.extend(test_data.y.detach().cpu().numpy())
        pred = model(test_data)
        test_pred_list.extend(pred.detach().cpu().numpy())

    train_result = mean_squared_error(train_label_list, train_pred_list)
    test_result = mean_squared_error(test_label_list, test_pred_list)

    print("train mse:", train_result, "test mse:", test_result)

if __name__ == '__main__':
    set_seed(100)
    model = Model().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    dataInitializer = DataInitializer(param)
    trainG, valG, testG = dataInitializer.createGraphList(param['window_size_K'])
    trainLoader = DataLoader(trainG, batch_size=param['batch_size'], shuffle=False)
    valLoader = DataLoader(valG)
    testLoader = DataLoader(testG)

    if param['TRAIN']:
        trainModel()
    if param['TEST']:
        testModel()
