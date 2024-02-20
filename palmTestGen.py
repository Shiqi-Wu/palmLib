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
import csv
import palmConfigGpu as pcg
import palmConfig as pc


def get_args_parser():
    parser = argparse.ArgumentParser(description="Parse command line arguments for testing.")
    
    # add -fName param
    parser.add_argument("--fName", type = str, help="File name to be processed", required=False, default = 'palmTestGen.yml')
    # add -gpu param
    parser.add_argument("--gpu", type = str, help = 'GPU ID to be used', required=False, default = "-1")

    args = parser.parse_args()
    return args

def generate_test_data(yaml_file='palmTestGen.yml', row_count=None):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    x_dim = config['PalmTestGen']['xDim']
    y_dim = config['PalmTestGen']['yDim']

    if row_count is None:
        row_count = config['PalmTestGen']['RowCount']

    csv_header = list(x_dim.keys()) + list(y_dim.keys())
    x_vec_list = []  # List to store x_vec for each row
    y_vec_list = []  # List to store y_vec for each row

    for _ in range(row_count):
        x_data = []
        for key, value in x_dim.items():
            x_min = value['min']
            x_max = value['max']
            x_step = value['step']
            x_value = math.floor(random.uniform(x_min, x_max) / x_step) * x_step
            x_data.append(x_value)

        y_data = []
        for key, value in y_dim.items():
            formula = value['formula']
            y_value = eval(formula, {'x1': x_data[0], 'x2': x_data[1]})  # Assuming x1 and x2 are the first two elements in x_data
            y_data.append(y_value)
        
        x_vec_list.append(x_data)
        y_vec_list.append(y_data)

    print("---CSV---")
    print(",".join(header))
    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        print(",".join(map(str, x_vec + y_vec)))
    write_csv('test_data.csv', header, x_vec_list, y_vec_list)

    return csv_header, x_vec_list, y_vec_list

def write_csv(file_name, header, x_vec_list, y_vec_list):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for x_vec, y_vec in zip(x_vec_list, y_vec_list):
            writer.writerow(x_vec + y_vec)

def test_functions(x_vec_list, y_vec_list):

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VecMax(x_vec)
        pcg_result = pcg.VecMax(x_vec)
        assert pc_result == pcg_result, f"VecMax({x_vec}) = {pc_result} != {pcg_result}"

        pc_result = pc.VecMin(x_vec)
        pcg_result = pcg.VecMin(x_vec)
        assert pc_result == pcg_result, f"VecMin({x_vec}) = {pc_result} != {pcg_result}"

        pc_result = pc.PositiveNumber(x_vec[0])
        pcg_result = pcg.PositiveNumber(x_vec[0])
        assert pc_result == pcg_result, f"PositiveNumber({x_vec[0]}) = {pc_result} != {pcg_result}"

        pc_result = pc.InSet(x_vec, x_vec[0])
        pcg_result = pcg.InSet(x_vec, x_vec[0])
        assert pc_result == pcg_result, f"InSet({x_vec}, {x_vec[0]}) = {pc_result} != {pcg_result}"

        pc_result = pc.InSet(x_vec, y_vec[0])
        pcg_result = pcg.InSet(x_vec, y_vec[0])
        assert pc_result == pcg_result, f"InSet({x_vec}, {y_vec[0]}) = {pc_result} != {pcg_result}"

        pc_result = pc.Contains(x_vec, x_vec)
        pcg_result = pcg.Contains(x_vec, x_vec)
        assert pc_result == pcg_result, f"Contains({x_vec}, {x_vec}) = {pc_result} != {pcg_result}"

        pc_result = pc.Contains(x_vec, y_vec)
        pcg_result = pcg.Contains(x_vec, y_vec)
        assert pc_result == pcg_result, f"Contains({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"

        pc_result = pc.VectorRange(x_vec)
        pcg_result = pcg.VectorRange(x_vec)
        assert pc_result == pcg_result, f"VectorRange({x_vec}) = {pc_result} != {pcg_result}"

        pc_result = pc.VectorCmpAllGt(x_vec, 0)
        pcg_result = pcg.VectorCmpAllGt(x_vec, 0)
        assert pc_result == pcg_result, f"VectorCmpAllGt({x_vec}, 0) = {pc_result} != {pcg_result}"

        pc_result = pc.VectorCmpAllLt(x_vec, 0)
        pcg_result = pcg.VectorCmpAllLt(x_vec, 0)
        assert pc_result == pcg_result, f"VectorCmpAllLt({x_vec}, 0) = {pc_result} != {pcg_result}"

        pc_result = pc.VectorCmp(x_vec, y_vec)
        pcg_result = pcg.VectorCmp(x_vec, y_vec)
        assert pc_result == pcg_result, f"VectorCmp({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"

        

    return

def main(args):
    if args.gpu == "-1":
        import palmConfig as pc
    else:
        import palmConfigGpu as pc


    # Test all functions in palmConfig.py or palmConfigGpu.py
    pc.test_all_functions()


if __name__ == '__main__':
    args = get_args_parser()

    header, x_vec_list, y_vec_list = generate_test_data(args.fName)
    
