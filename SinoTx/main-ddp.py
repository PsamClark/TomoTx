#! /home/iib14472/.conda/envs/wedgesketch/bin/python
import torch, argparse, os, time, sys, shutil, yaml
from data_handler import prepare_training_data
import numpy as np
from train import train_main
import mrcfile

parser = argparse.ArgumentParser(description='SinoTx')
parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-cfg',    type=str, required=True, help='path to config yaml file')
parser.add_argument('-df',    type=str, required=True, help='path to datafiles')
parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main(args):

    # training task init
    params = yaml.load(open(args.cfg, 'r'), Loader=yaml.CLoader)
    result = train_main(trainfile=args.df,params = params,
                        vsplit=0.7,
                      tsplit=0.1, distributed=False)

    with mrcfile.new('output.mrc',overwrite=True) as mf:
        mf.set_data(result.detach().numpy())

if __name__ == "__main__":
    args, unparsed = parser.parse_known_args()



    main(args)
