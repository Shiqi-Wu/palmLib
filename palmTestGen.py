import yaml
import sys
import random
import math
import os
import argparse
import palmConfig as pc
import datetime
import json
import os
import time

def get_args_parser():
    parser = argparse.ArgumentParser(description="Parse command line arguments for testing.")
    
    # add -fName param
    parser.add_argument("--fName", type = str, help="File name to be processed", required=False, default = 'palmTestGen.yml')
    # add -gpu param
    parser.add_argument("--gpu", type = str, help = 'GPU ID to be used', required=False, default = "-1")

    args = parser.parse_args()
    return args

def main(args):

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

# Use cpu/gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def formatTemp(temp):
    return eval(f"f'{temp}'")

# print (f'len(sys.argv)={len(sys.argv)}')

fName = 'palmTestGen.yml'
if len(sys.argv) > 1:
    fName = str(sys.argv[1])

print (f'palmTestGen:{fName}')

with open(fName, 'r') as nFile:
    nRoot = yaml.safe_load(nFile)

print(f"Load Success:{nRoot['PalmTestGen']['xDim']['x1']}")

palmTestGen = nRoot['PalmTestGen']
print(f"Row Count: {palmTestGen['RowCount']}")
print(f"xDim : {len(palmTestGen['xDim'])}")
print(f"yDim : {len(palmTestGen['yDim'])}")

cnt = palmTestGen['RowCount']
if len(sys.argv) > 2: # override generate how many rows
    cnt = int(sys.argv[2])
    print(f"Row Count Override to:{cnt}")

xDim = palmTestGen['xDim']
yDim = palmTestGen['yDim']

# Create xExecString
xGen = ""
xVar = ''
csvHeader = ''
# print(f"type(xDim) = {type(xDim)}")
for kk,vv in xDim.items():
    # print(f"xVar = {kk}")
    # print(f"type(vv) = {type(vv)}")
    xMin = vv['min']
    xMax = vv['max']
    xStep = vv['step']
    xGen += f"{kk} = math.floor(random.uniform({xMin}, {xMax})/{xStep})*{xStep};"
    if len(xVar) > 1:
        xVar += ','
        csvHeader += ','
    xVar += '{' + f"{kk}" + '}'
    csvHeader += kk
# print(xGen)       
# print(xVar)
# exec(xGen)

yGen = ''
for kk, vv in yDim.items():
    yGen += f"{kk} = {vv};"
    xVar += ',{' + f"{kk}" + '}'
    csvHeader += f",{kk}"
# exec(yGen)

# print(formatTemp(xVar))
# print(xVar)
print("---CSV---")
print(csvHeader)
for ii in range(cnt):
    exec(xGen)
    exec(yGen)
    print(formatTemp(xVar))

## ------------Test for library----------------

# Test VecMax Function
print("Testing VecMax:")