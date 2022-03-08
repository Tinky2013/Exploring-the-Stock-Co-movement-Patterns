import pandas as pd
import numpy as np

adj = np.load('preprocess_data/edges/weight_train_12K_0.npy')
print(adj)
print(type(adj))
print(adj.shape)