# Application of CSDI in Finance

## Introduction
This is a student demo project to impute missing historical data of stocks using a diffusion model introduced in [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502).

## How to use
1. Put your csv file in the ```/input``` folder and rename it as ```missing.csv```. Example file is provided.
2. Use your own ```data_preprocess``` program to generate a file called ```missing_processed.csv``` in the ```/input``` folder. You can refer to the ```data_preprocess_demo.py``` for more detailed infomation. Example file is also provided.
3. Run ```exe_impute.py``` with proper arguments (```--modelfolder``` is a must and others are optional).
    - ```--device```: choose the GPU you want to use (default ```cuda:0```)
    - ```--unconditional```: we won't use it in the imputation and it can be ignored
    - ```--modelfolder```: the pretrained model parameters you want to use (they are stored in the ```/save``` folder, the number means missing rate, default ```0.1missing```)
    - ```--nsample```: number of generated samples for taking average (default ```100```)
4. Run your own ```dataframe_build``` program and generate a filled csv file called ```filled.csv``` in the ```/output``` folder. You can refer to the ```dataframe_build_demo.py``` for more detailed infomation.

## Example
```
python3 exe_impute.py --device cuda:0 --modelfolder 0.3missing --nsample 100
```

## Notice
Here we only trained the model for daily minute-level data imputation for A-stock in China at different missing rate, which means the number of the rows of data for one date is fixed at **241** (9:30 - 11:30 and 13:00 - 15:00) and the number of the columns is fixed at **14** (14 different indexes, as shown in the example files).

You can train your own model of different settings using the ```exe_physio``` program.

More details of the original CSDI can be found [here](https://github.com/ermongroup/CSDI).