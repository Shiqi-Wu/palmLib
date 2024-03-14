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
import numpy as np

err = 1e-8

def get_args_parser():
    parser = argparse.ArgumentParser(description="Parse command line arguments for testing.")
    
    # add -fName param
    parser.add_argument("--fName", type = str, help="File name to be processed", required=False, default = 'palmTestGen.yml')
    # add -gpu param
    parser.add_argument("--gpu", type = str, help = 'GPU ID to be used', required=False, default = "-1")
    parser.add_argument("--row_count", type = int, help = 'Number of rows in the test data', required=False, default = 10)
    parser.add_argument("--dim", type = int, help = 'Dimension of the test data', required=False, default = 10)

    args = parser.parse_args()
    return args

def generate_test_data(yaml_file='palmTestGen.yml'):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    x_dim = config['PalmTestGen']['xDim']
    y_dim = config['PalmTestGen']['yDim']

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
    print(",".join(csv_header))
    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        print(",".join(map(str, x_vec + y_vec)))
    write_csv('test_data.csv', csv_header, x_vec_list, y_vec_list)

    return csv_header, x_vec_list, y_vec_list


def generate_test_data_random(row_count = 10, dim = 10, iMin = -10, iMax = 10):
    csv_header = [f'x{i}' for i in range(dim)] + [f'y{i}' for i in range(dim)] + [f'wm{i}{j}' for i in range(dim) for j in range(dim)]
    x_vec_list = []  # List to store x_vec for each row
    y_vec_list = []  # List to store y_vec for each row
    wm_vec_list = []

    for _ in range(row_count):
        x_data = [random.uniform(iMin, iMax) for _ in range(dim)]
        y_data = [random.uniform(iMin, iMax) for _ in range(dim)]
        wm_data = [[random.uniform(iMin, iMax) for _ in range(dim)] for _ in range(dim)]
        x_vec_list.append(x_data)
        y_vec_list.append(y_data)
        wm_vec_list.append(wm_data)

    # print("---CSV---")
    # print(",".join(csv_header))
    # for x_vec, y_vec in zip(x_vec_list, y_vec_list):
    #     print(",".join(map(str, x_vec + y_vec)))
    # write_csv('test_data.csv', csv_header, x_vec_list, y_vec_list, wm_vec_list)

    return csv_header, x_vec_list, y_vec_list, wm_vec_list

def write_csv(file_name, header, x_vec_list, y_vec_list):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for x_vec, y_vec in zip(x_vec_list, y_vec_list):
            writer.writerow(x_vec + y_vec)

def read_csv(file_name):
    with open(file_name, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        x_vec_list = []
        y_vec_list = []
        for row in reader:
            x_vec = list(map(float, row[:len(row)//2]))
            y_vec = list(map(float, row[len(row)//2:]))
            x_vec_list.append(x_vec)
            y_vec_list.append(y_vec)
    return header, x_vec_list, y_vec_list

def test_palmConfig():
    # Common vectors used in tests
    vec123 = [1, 2, 3]
    vec456 = [4, 5, 6]
    
    # Test InfoLevel enum
    assert pc.InfoLevel.TRACE.value == 0
    assert pc.InfoLevel.DEBUG.value == 1
    assert pc.InfoLevel.ERROR.value == 2
    assert pc.InfoLevel.WARN.value == 3
    assert pc.InfoLevel.INFO.value == 4

    # Test Assert function
    assert pc.Assert(True, "Test Assert") == True
    assert pc.AssertWarn(True, "Test AssertWarn") == True

    # Test VecMax function
    assert pc.VecMax(vec123, vec456) == vec456

    # Test VecMin function
    assert pc.VecMin(vec123, vec456) == vec123

    # Test PositiveNumber function
    assert pc.PositiveNumber(1) == True

    # Test InSet function
    assert pc.InSet(vec123, 2) == True
    assert pc.InSet(vec123, 4) == False

    # Test Contains function
    assert pc.Contains(vec123, [2, 3]) == True
    assert pc.Contains(vec123, [4, 5]) == False

    # Test VectorRange function
    assert pc.VectorRange(vec123) == (1, 3)
    assert pc.VectorRange(vec123, iMin=2, iMax=2) == (2, 2)

    # Test VectorCmpAllGt function
    assert pc.VectorCmpAllGt(vec123, 0) == True
    assert pc.VectorCmpAllGt(vec123, 2) == False

    # Test VectorCmpAllLt function
    assert pc.VectorCmpAllLt(vec123, 4) == True
    assert pc.VectorCmpAllLt(vec123, 2) == False

    # Test VectorCmp function
    assert pc.VectorCmp(vec123, vec456) == -1
    assert pc.VectorCmp(vec456, vec123) == 1
    assert pc.VectorCmp(vec123, vec123) == 0

    # Test VectorAdd function
    assert pc.VectorAdd(vec123, vec456) == [5, 7, 9]

    # Test VectorSub function
    assert pc.VectorSub(vec123, vec456) == [-3, -3, -3]

    # Test VectorMul function
    assert pc.VectorMul(vec123, vec456) == [4, 10, 18]

    # Test Normalize function
    assert pc.Normalize(vec123, 0, 10) == [0.1, 0.2, 0.3]

    # Test Denormalize function
    assert pc.Denormalize([0.1, 0.2, 0.3], 0, 10) == vec123

    # Test VectorSum function
    assert pc.VectorSum(vec123) == 6

    # Test VectorDotMul function
    assert pc.VectorDotMul(vec123, vec456) == 32

    # Test VectorABS function
    assert pc.VectorABS([3, 4]) == 5

    # Test VectorSMul function
    assert pc.VectorSMul(vec123, 2) == [2, 4, 6]

    # Test VectorCoss function
    assert pc.VectorCoss(vec123, vec456, 2) == 16

    # Test VectorEuclideanD function
    assert pc.VectorEuclideanD(vec123, vec456) == math.sqrt(27)

    # Test VectorManhattanD function
    assert pc.VectorManhattanD(vec123, vec456) == 9

    # Test VectorCosins function
    assert pc.VectorCosins(vec123, vec456) == 0.9746318461970762

 # Test NormalVector functio    
    wm = np.array([[1, 2], [3, 4]])
    bb = np.array([5, 6])
    assert np.allclose(pc.NormalVector(wm, bb), np.array([-4.0, 4.5]))

    # Test Point2PlaneDistance function
    pt = [1, 2, 3]
    nv = [4, 5, 6]
    bb = 7
    assert pc.Point2PlaneDistance(pt, nv, bb) == 2.8490144114909484

    # Test LUDecomp function
    mat = [[3, 4], [6, 13]]
    LUL, LUU = pc.LUDecomp(mat)
    assert np.allclose(LUL, [[1, 0], [2, 1]])
    assert np.allclose(LUU, [[3, 4], [0, 5]])

    # Test MatrixReduce function
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    reduced_mat, reduced_rowName, reduced_colName = pc.MatrixReduce(mat, rowName, colName)
    assert np.allclose(reduced_mat, [[1, 2], [3, 4]])
    assert reduced_rowName == ['row1', 'row2']
    assert reduced_colName == ['col1', 'col2']

    # Test MatrixMergeRow function
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    mergeIdxList = [0, 1]
    merged_mat, merged_rowName, merged_colName = pc.MatrixMergeRow(mat, rowName, colName, mergeIdxList)
    assert np.allclose(merged_mat, [[1, 2], [3, 4]])
    assert merged_rowName == ['row1', 'row2']
    assert merged_colName == ['col1', 'col2']

    print("All functions tested successfully!")


def test_pc_vs_pcg(x_vec_list, y_vec_list, wm_vec_list):

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VecMax(x_vec, y_vec)
        pcg_result = pcg.VecMax(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"VecMax({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VecMax pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VecMin(x_vec, y_vec)
        pcg_result = pcg.VecMin(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"VecMin({x_vec}) = {pc_result} != {pcg_result}"
    print('VecMin pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.PositiveNumber(x_vec[0])
        pcg_result = pcg.PositiveNumber(x_vec[0])
        assert pc_result == pcg_result, f"PositiveNumber({x_vec[0]}) = {pc_result} != {pcg_result}"
    print('PositiveNumber pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.InSet(x_vec, x_vec[0])
        pcg_result = pcg.InSet(x_vec, x_vec[0])
        assert pc_result == pcg_result.tolist(), f"InSet({x_vec}, {x_vec[0]}) = {pc_result} != {pcg_result}"
    print('InSet pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.InSet(x_vec, y_vec[0])
        pcg_result = pcg.InSet(x_vec, y_vec[0])
        assert pc_result == pcg_result.tolist(), f"InSet({x_vec}, {y_vec[0]}) = {pc_result} != {pcg_result}"
    print('InSet pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.Contains(x_vec, x_vec)
        pcg_result = pcg.Contains(x_vec, x_vec)
        assert pc_result == pcg_result.tolist(), f"Contains({x_vec}, {x_vec}) = {pc_result} != {pcg_result}"
    print('Contains pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.Contains(x_vec, y_vec)
        pcg_result = pcg.Contains(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"Contains({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('Contains pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorRange(x_vec)
        pcg_result = pcg.VectorRange(x_vec)
        assert pc_result == pcg_result, f"VectorRange({x_vec}) = {pc_result} != {pcg_result}"
    print('VectorRange pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorCmpAllGt(x_vec, 0)
        pcg_result = pcg.VectorCmpAllGt(x_vec, 0)
        assert pc_result == pcg_result.tolist(), f"VectorCmpAllGt({x_vec}, 0) = {pc_result} != {pcg_result}"
    print('VectorCmpAllGt pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorCmpAllLt(x_vec, 0)
        pcg_result = pcg.VectorCmpAllLt(x_vec, 0)
        assert pc_result == pcg_result.tolist(), f"VectorCmpAllLt({x_vec}, 0) = {pc_result} != {pcg_result}"
    print('VectorCmpAllLt pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorCmp(x_vec, y_vec)
        pcg_result = pcg.VectorCmp(x_vec, y_vec)
        assert pc_result == pcg_result, f"VectorCmp({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorCmp pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorAdd(x_vec, y_vec)
        pcg_result = pcg.VectorAdd(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"VectorAdd({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorAdd pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorSub(x_vec, y_vec)
        pcg_result = pcg.VectorSub(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"VectorSub({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorSub pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorMul(x_vec, y_vec)
        pcg_result = pcg.VectorMul(x_vec, y_vec)
        assert pc_result == pcg_result.tolist(), f"VectorMul({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorMul pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.Normalize(x_vec, 0, 10)
        pcg_result = pcg.Normalize(x_vec, 0, 10)
        assert pc_result == pcg_result.tolist(), f"Normalize({x_vec}, 0, 10) = {pc_result} != {pcg_result}"
    print('Normalize pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.Denormalize(x_vec, 0, 10)
        pcg_result = pcg.Denormalize(x_vec, 0, 10)
        assert pc_result == pcg_result.tolist(), f"Denormalize({x_vec}, 0, 10) = {pc_result} != {pcg_result}"
    print('Denormalize pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorSum(x_vec)
        pcg_result = pcg.VectorSum(x_vec)
        assert abs(pc_result - pcg_result) <err, f"VectorSum({x_vec}) = {pc_result} != {pcg_result}"
    print('VectorSum pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorDotMul(x_vec, y_vec)
        pcg_result = pcg.VectorDotMul(x_vec, y_vec)
        assert abs(pc_result - pcg_result) < err, f"VectorDotMul({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorDotMul pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorABS(x_vec)
        pcg_result = pcg.VectorABS(x_vec)
        assert abs(pc_result - pcg_result) < err, f"VectorABS({x_vec}) = {pc_result} != {pcg_result}"
    print('VectorABS pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorSMul(x_vec, 2)
        pcg_result = pcg.VectorSMul(x_vec, 2)
        assert pc_result == pcg_result.tolist(), f"VectorSMul({x_vec}, 2) = {pc_result} != {pcg_result}"
    print('VectorSMul pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorCoss(x_vec, y_vec, 2)
        pcg_result = pcg.VectorCoss(x_vec, y_vec, 2)
        assert abs(pc_result - pcg_result) < err, f"VectorCoss({x_vec}, {y_vec}, 2) = {pc_result} != {pcg_result}"
    print('VectorCoss pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorEuclideanD(x_vec, y_vec)
        pcg_result = pcg.VectorEuclideanD(x_vec, y_vec)
        assert abs(pc_result - pcg_result) < err, f"VectorEuclideanD({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorEuclideanD pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorManhattanD(x_vec, y_vec)
        pcg_result = pcg.VectorManhattanD(x_vec, y_vec)
        assert abs(pc_result - pcg_result) < err, f"VectorManhattanD({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorManhattanD pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.VectorCosins(x_vec, y_vec)
        pcg_result = pcg.VectorCosins(x_vec, y_vec)
        assert abs(pc_result - pcg_result) < err, f"VectorCosins({x_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('VectorCosins pass!')

    for wm_vec, y_vec in zip(wm_vec_list, y_vec_list):
        pc_result = pc.NormalVector(wm_vec, y_vec)
        pcg_result = pcg.NormalVector(wm_vec, y_vec)
        assert np.allclose(pc_result, pcg_result.tolist()), f"NormalVector({wm_vec}, {y_vec}) = {pc_result} != {pcg_result}"
    print('NormalVector pass!')

    for x_vec, y_vec in zip(x_vec_list, y_vec_list):
        pc_result = pc.Point2PlaneDistance(x_vec, y_vec, 2)
        pcg_result = pcg.Point2PlaneDistance(x_vec, y_vec, 2)
        assert abs(pc_result - pcg_result) < err, f"Point2PlaneDistance({x_vec}, {y_vec}, 2) = {pc_result} != {pcg_result}"
    print('Point2PlaneDistance pass!')

    for wm_vec in wm_vec_list:
        # print(len(wm_vec))
        # print(len(wm_vec[0]))
        # print(wm_vec[0][0])
        pc_result = pc.LUDecomp(wm_vec)
        pcg_result = pcg.LUDecomp(wm_vec)
        assert np.allclose(pc_result, pcg_result), f"LUDecomp({wm_vec}) = {pc_result} != {pcg_result}"
    print('LUDecomp pass!')

    print("All functions tested successfully!")


    return


if __name__ == '__main__':
    args = get_args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    header, x_vec_list, y_vec_list, wm_vec_list = generate_test_data_random(row_count = args.row_count, dim = args.dim)
    test_palmConfig()
    test_pc_vs_pcg(x_vec_list, y_vec_list, wm_vec_list)

    print("All tests passed!")
    
