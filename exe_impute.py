import argparse
import torch
import yaml

from main_model import CSDI_Physio
from dataset_impute import get_impute_dataloader
from utils import predict

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="0.1missing")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

# load config
path = "config/base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)
config["model"]["is_unconditional"] = args.unconditional

# prepare dataset
impute_loader = get_impute_dataloader()

# init model
model = CSDI_Physio(config, args.device).to(args.device)
model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

# imputation
predict(model, impute_loader, nsample=args.nsample, scaler=1, foldername="output/")
