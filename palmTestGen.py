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
    assert InfoLevel.ERROR.value == 2
    assert InfoLevel.WARN.value == 3
    assert InfoLevel.INFO.value == 4

    # Test Assert function
    assert Assert(True, "Test Assert") == True
    assert AssertWarn(True, "Test AssertWarn") == True

    # Test VecMax function
    assert VecMax(vec123, vec456) == vec456

    # Test VecMin function
    assert VecMin(vec123, vec456) == vec123

    # Test SetRunningLevel function
    SetRunningLevel(InfoLevel.DEBUG)
    assert RunningLevel == InfoLevel.DEBUG

    # Test PositiveNumber function
    assert PositiveNumber(1) == True

    # Test InSet function
    assert InSet(vec123, 2) == True
    assert InSet(vec123, 4) == False

    # Test Contains function
    assert Contains(vec123, [2, 3]) == True
    assert Contains(vec123, [4, 5]) == False

    # Test VectorRange function
    assert VectorRange(vec123) == (1, 3)
    assert VectorRange(vec123, iMin=0, iMax=10) == (0, 10)

    # Test VectorCmpAllGt function
    assert VectorCmpAllGt(vec123, 0) == True
    assert VectorCmpAllGt(vec123, 2) == False

    # Test VectorCmpAllLt function
    assert VectorCmpAllLt(vec123, 4) == True
    assert VectorCmpAllLt(vec123, 2) == False

    # Test VectorCmp function
    assert VectorCmp(vec123, vec456) == -1
    assert VectorCmp(vec456, vec123) == 1
    assert VectorCmp(vec123, vec123) == 0

    # Test VectorAdd function
    assert VectorAdd(vec123, vec456) == [5, 7, 9]

    # Test VectorSub function
    assert VectorSub(vec123, vec456) == [-3, -3, -3]

    # Test VectorMul function
    assert VectorMul(vec123, vec456) == [4, 10, 18]

    # Test Normalize function
    assert Normalize(vec123, 0, 10) == [0.1, 0.2, 0.3]

    # Test Denormalize function
    assert Denormalize([0.1, 0.2, 0.3], 0, 10) == vec123

    # Test VectorSum function
    assert VectorSum(vec123) == 6

    # Test VectorDotMul function
    assert VectorDotMul(vec123, vec456) == 32

    # Test VectorABS function
    assert VectorABS([3, 4]) == 5

    # Test VectorSMul function
    assert VectorSMul(vec123, 2) == [2, 4, 6]

    # Test VectorCoss function
    assert VectorCoss(vec123, vec456, 2) == 14

    # Test VectorEuclideanD function
    assert VectorEuclideanD(vec123, vec456) == math.sqrt(27)

    # Test VectorManhattanD function
    assert VectorManhattanD(vec123, vec456) == 9

    # Test VectorCosins function
    assert VectorCosins(vec123, vec456) == 0.9746318461970762

 # Test NormalVector function
    wm = np.array([[1, 2], [3, 4]])
    bb = np.array([5, 6])
    assert np.allclose(NormalVector(wm, bb), np.array([-4, 4]))

    # Test Point2PlaneDistance function
    pt = [1, 2, 3]
    nv = [4, 5, 6]
    bb = 7
    assert Point2PlaneDistance(pt, nv, bb) == -4.898979485566356

    # Test LUDecomp function
    mat = [[1, 2], [3, 4]]
    LUL, LUU = LUDecomp(mat)
    assert np.allclose(LUL, [[1, 0], [3, -2]])
    assert np.allclose(LUU, [[1, 2], [0, -2]])

    # Test MatrixReduce function
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    reduced_mat, reduced_rowName, reduced_colName = MatrixReduce(mat, rowName, colName)
    assert np.allclose(reduced_mat, [[1, 2], [3, 4]])
    assert reduced_rowName == ['row1', 'row2']
    assert reduced_colName == ['col1', 'col2']

    # Test MatrixMergeRow function
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    mergeIdxList = [0, 1]
    merged_mat, merged_rowName, merged_colName = MatrixMergeRow(mat, rowName, colName, mergeIdxList)
    assert np.allclose(merged_mat, [[1, 2], [3, 4]])
    assert merged_rowName == ['row1', 'row2']
    assert merged_colName == ['col1', 'col2']

    print("All functions tested successfully!")


def test_pc_vs_pcg(x_vec_list, y_vec_list):

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


if __name__ == '__main__':
    args = get_args_parser()

    header, x_vec_list, y_vec_list = generate_test_data(args.fName)
    test_palmConfig()
    test_pc_vs_pcg(x_vec_list, y_vec_list)

    print("All tests passed!")
    
