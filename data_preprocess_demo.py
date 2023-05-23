import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime


#Pre-preprocess
df = pd.read_csv('./input/missing.csv')
df['date'] = df['date_time'].str[0:11]
df.index = df['date']
df.replace(0, np.nan, inplace=True)
#df = df.drop(['date_time','code','Unnamed: 0','date','raise_num','fall_num'], axis=1)
df = df.drop(['date_time','code','date','raise_num','fall_num'], axis=1)
df.to_csv('./input/missing_processed.csv')