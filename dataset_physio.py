import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

class Physio_Dataset(Dataset):
    
    def __init__(self, eval_length = 241, use_index_list = None, missing_ratio = 0.1, seed = 0):
        self.eval_length = eval_length
        np.random.seed(seed)
        
        df = pd.read_csv('',index_col = 'date')  # full data
        df_gt = pd.read_csv('',index_col = 'date')  # missing data
        n_columns = len(df.columns)
        
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./data/physio_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )
        
        date = df.index.unique()
        for i in date:
            observed_values = np.array(df.loc[[i]])
            observed_masks = ~np.isnan(observed_values) # take the negation of np.isnan
            gt_masks = ~np.isnan(np.array(df_gt.loc[[i]]))

            observed_values = np.nan_to_num(observed_values) 
            observed_masks = observed_masks.astype("float32") 
            gt_masks = gt_masks.astype("float32") 
        
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)
        
        # three-day ver. testing
        # for i in range(0,597,3):  # FIXME hard code
        #     observed_values = np.array(df.loc[date[i]:date[i+2]])
        #     observed_masks = ~np.isnan(observed_values) # take the negation of np.isnan
        #     gt_masks = ~np.isnan(np.array(df_gt.loc[date[i]:date[i+2]]))

        #     observed_values = np.nan_to_num(observed_values) 
        #     observed_masks = observed_masks.astype("float32") 
        #     gt_masks = gt_masks.astype("float32") 
        
        #     self.observed_values.append(observed_values)
        #     self.observed_masks.append(observed_masks)
        #     self.gt_masks.append(gt_masks)
            
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
        ms_path = ""
        with open(ms_path, "wb") as f:
            pickle.dump([mean, std], f)

        self.observed_values = (
            (self.observed_values - mean)/std * self.observed_masks
        )
        
        with open(path, "wb") as f:
            pickle.dump(
                [self.observed_values,self.observed_masks,self.gt_masks], f
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


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = Physio_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    # train_start = 0
    # valid_start = int(0.8 * len(dataset))
    # test_start = int(0.9 * len(dataset))
    # train_index = indlist[train_start:valid_start]
    # valid_index = indlist[valid_start:test_start]
    # test_index = indlist[test_start:]

    dataset = Physio_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Physio_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Physio_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    dataset = Physio_Dataset(missing_ratio = 0.1,seed = 1)
    print(np.arange(len(dataset)))
    train_loader = DataLoader(dataset,batch_size = 100,shuffle = 1)
    print(next(iter(train_loader))['observed_data'])
    print(next(iter(train_loader))['observed_mask'].size())
    print(next(iter(train_loader))['gt_mask'].size())
    print(next(iter(train_loader))['timepoints'].size())