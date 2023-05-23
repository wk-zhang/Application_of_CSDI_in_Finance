import torch
import pickle
import pandas as pd

def get_quantile(samples,q,dim=1):
    return torch.quantile(samples,q,dim=dim).cpu().numpy()

nsample = 100 # number of generated sample
path = './output/generated_outputs_nsample' + str(nsample) + '.pk' 
with open(path, 'rb') as f:
    samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load(f)

all_target_np = all_target.cpu().numpy()
all_evalpoint_np = all_evalpoint.cpu().numpy()
all_observed_np = all_observed.cpu().numpy()
all_given_np = all_observed_np - all_evalpoint_np

K = samples.shape[-1] #feature
L = samples.shape[-2] #time length


path = './output/impute_meanstd.pk'
with open(path, 'rb') as f:
    train_mean,train_std = pickle.load(f)
all_target_np=(all_target_np*train_std+train_mean)

samples=(samples.cpu()*train_std+train_mean)

qlist =[0.05,0.25,0.5,0.75,0.95]
quantiles_imp= []
for q in qlist:
    quantiles_imp.append(get_quantile(samples, q, dim=1)*(1-all_given_np) + all_target_np * all_given_np)

# Here we reconstruct the imputed dataframe
df = pd.read_csv('./input/missing.csv')  # input a sample csv format
final_col = df.columns
result = pd.DataFrame(columns=final_col)
imp_col = ['open', 'high', 'low', 'close', 'avgPrice',
       'volume', 'money', 'trans_num',
       'possitive_buy_large_vol', 'possitive_sell_large_vol',
       'possitive_buy_main_vol', 'possitive_sell_main_vol',
       'possitive_buy_middle_vol', 'possitive_sell_middle_vol']
K = len(imp_col)
df['date'] = df['date_time'].str[0:11]
date = df['date'].unique()
dataind = len(set(date))

for dataind in range(dataind):
    date = df['date']
    temp = pd.DataFrame(columns=final_col)
    for k in range(K):
        temp[imp_col[k]] = quantiles_imp[2][dataind,:,k]
    temp['code'] = df['code'].loc[df['date'] == date[dataind]]
    temp['date_time'] = df['date_time'].loc[df['date'] == date[dataind]]
    result = pd.concat([result,temp])
result.to_csv('./output/filled.csv')