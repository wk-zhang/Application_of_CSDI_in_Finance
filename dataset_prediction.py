import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

class Prediction_Dataset(Dataset):
    
    def __init__(self, eval_length = 241, use_index_list = None):
        self.eval_length = eval_length
        
        df = pd.read_csv('',index_col = 'date')  # full data
        df_gt = pd.read_csv('',index_col = 'date')  # missing data
        n_columns = len(df.columns)
        
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        
        date = df.index.unique()
        for i in date:
            observed_values = np.array(df.loc[[i]])
            observed_masks = ~np.isnan(observed_values) # take the negation of np.isnan
            gt_masks = ~np.isnan(np.array(df_gt.loc[[i]]))

            # full all NaN with 0 after setting up masks
            observed_values = np.nan_to_num(observed_values) 
            observed_masks = observed_masks.astype("float32") 
            gt_masks = gt_masks.astype("float32") 
        
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)
            
        self.observed_values = np.array(self.observed_values)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        
        
        tmp_values = self.observed_values.reshape(-1,n_columns)
        tmp_masks = self.observed_masks.reshape(-1,n_columns)
        mean = np.zeros(n_columns)
        std = np.zeros(n_columns)
        for k in range(n_columns):
            c_data = tmp_values[:, k][tmp_masks[:, k]==1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()

        # store mean & std for visualization
        ms_path = "/home/sida/zwk/Application_of_CSDI_in_Finance/data/predict_meanstd.pk"
        with open(ms_path, "wb") as f:
            pickle.dump([mean, std], f)

        self.observed_values = (
            (self.observed_values - mean)/std * self.observed_masks
        )
        
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list
            
    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_predict_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    dataset = Prediction_Dataset()
    predict_loader = DataLoader(dataset)
    
    return predict_loader
