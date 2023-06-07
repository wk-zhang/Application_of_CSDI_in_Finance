import numpy as np
import pandas as pd


# read the original missing.csv
df = pd.read_csv('./input/missing.csv')

# we only use date in the imputation process
# (consider data in the same date as a whole)
# remember there should be 241 rows for each date
df['date'] = df['date_time'].str[0:11]
df.index = df['date']  # set date column as the index
# if zero data means invalid data, convert it to NaN
df.replace(0, np.nan, inplace=True)

# be aware of some auto-generated columns, no-need-to-impute columns, and remember to drop original date_time column
df = df.drop(['date_time', 'stock_code', 'Unnamed: 0'], axis=1)

# save it in the /input folder as missing_processed.csv
df.to_csv('./input/missing_processed.csv')