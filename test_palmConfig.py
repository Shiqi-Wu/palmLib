import numpy as np
import math

def test_InfoLevel():
    assert InfoLevel.TRACE.value == 0
    assert InfoLevel.DEBUG.value == 1
    assert InfoLevel.ERROR.value == 2
    assert InfoLevel.WARN.value == 3
    assert InfoLevel.INFO.value == 4

def test_Assert():
    assert Assert(True, "Test Assert") == True
    assert Assert(False, "Test Assert") == False

def test_AssertWarn():
    assert AssertWarn(True, "Test AssertWarn") == True
    assert AssertWarn(False, "Test AssertWarn") == False

def test_VecMax():
    aVec = [1, 2, 3]
    bVec = [4, 5, 6]
    assert VecMax(aVec, bVec) == [4, 5, 6]

def test_VecMin():
    aVec = [1, 2, 3]
    bVec = [4, 5, 6]
    assert VecMin(aVec, bVec) == [1, 2, 3]

def test_Print():
    # Since the Print function only prints to the console, it's difficult to write a test for it.
    # You can manually verify the output when running the code.

def test_SetRunningLevel():
    SetRunningLevel(InfoLevel.DEBUG)
    assert RunningLevel == InfoLevel.DEBUG

def test_PositiveNumber():
    assert PositiveNumber(1) == True
    assert PositiveNumber(-1) == False

def test_InSet():
    listX = [1, 2, 3]
    assert InSet(listX, 1) == True
    assert InSet(listX, 4) == False

def test_Contains():
    list1 = [1, 2, 3]
    list2 = [1, 2]
    assert Contains(list1, list2) == True
    assert Contains(list2, list1) == False

def test_VectorRange():
    vv = [1, 2, 3]
    assert VectorRange(vv) == (1, 3)
    assert VectorRange(vv, 0, 10) == (0, 10)

def test_VectorCmpAllGt():
    vv = [1, 2, 3]
    assert VectorCmpAllGt(vv, 0) == True
    assert VectorCmpAllGt(vv, 2) == False

def test_VectorCmpAllLt():
    vv = [1, 2, 3]
    assert VectorCmpAllLt(vv, 4) == True
    assert VectorCmpAllLt(vv, 2) == False

def test_VectorCmp():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorCmp(vv1, vv2) == -1
    assert VectorCmp(vv2, vv1) == 1
    assert VectorCmp(vv1, vv1) == 0

def test_VectorAdd():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorAdd(vv1, vv2) == [5, 7, 9]

def test_VectorSub():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorSub(vv1, vv2) == [-3, -3, -3]

def test_VectorMul():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorMul(vv1, vv2) == [4, 10, 18]

def test_Normalize():
    vv = [1, 2, 3]
    assert Normalize(vv, 0, 10) == [0.1, 0.2, 0.3]

def test_Denormalize():
    vv = [0.1, 0.2, 0.3]
    assert Denormalize(vv, 0, 10) == [1, 2, 3]

def test_VectorSum():
    vv = [1, 2, 3]
    assert VectorSum(vv) == 6

def test_VectorDotMul():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorDotMul(vv1, vv2) == 32

def test_VectorABS():
    vv = [1, 2, 3]
    assert VectorABS(vv) == math.sqrt(14)

def test_VectorSMul():
    vv = [1, 2, 3]
    assert VectorSMul(vv, 2) == [2, 4, 6]

def test_VectorCoss():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorCoss(vv1, vv2, 2) == 8

def test_VectorEuclideanD():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorEuclideanD(vv1, vv2) == math.sqrt(27)

def test_VectorManhattanD():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorManhattanD(vv1, vv2) == 9

def test_VectorCosins():
    vv1 = [1, 2, 3]
    vv2 = [4, 5, 6]
    assert VectorCosins(vv1, vv2) == 0.9746318461970762

def test_NormalVector():
    wm = np.array([[1, 2], [3, 4]])
    bb = np.array([5, 6])
    assert np.allclose(NormalVector(wm, bb), np.array([-4. ,  4.5]))

def test_Point2PlaneDistance():
    pt = [1, 2, 3]
    nv = [4, 5, 6]
    bb = 7
    assert Point2PlaneDistance(pt, nv, bb) == -1.0

def test_LUDecomp():
    mat = [[1, 2], [3, 4]]
    LUL, LUU = LUDecomp(mat)
    assert np.allclose(LUL, [[1, 0], [3, -2]])
    assert np.allclose(LUU, [[1, 2], [0, -2]])

def test_MatrixReduce():
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    assert MatrixReduce(mat, rowName, colName) == (mat, rowName, colName)

def test_MatrixMergeRow():
    mat = [[1, 2], [3, 4]]
    rowName = ['row1', 'row2']
    colName = ['col1', 'col2']
    mergeIdxList = [0, 1]
    assert MatrixMergeRow(mat, rowName, colName, mergeIdxList) == (mat, rowName, colName)

def test_PrintMatrix():
    # Since the PrintMatrix function only prints to the console, it's difficult to write a test for it.
    # You can manually verify the output when running the code.

def run_tests():
    test_InfoLevel()
    test_Assert()
    test_AssertWarn()
    test_VecMax()
    test_VecMin()
    test_Print()
    test_SetRunningLevel()
    test_PositiveNumber()
    test_InSet()
    test_Contains()
    test_VectorRange()
    test_VectorCmpAllGt()
    test_VectorCmpAllLt()
    test_VectorCmp()
    test_VectorAdd()
    test_VectorSub()
    test_VectorMul()
    test_Normalize()
    test_Denormalize()
    test_VectorSum()
    test_VectorDotMul()
    test_VectorABS()
    test_VectorSMul()
    test_VectorCoss()
    test_VectorEuclideanD()
    test_VectorManhattanD()
    test_VectorCosins()
    test_NormalVector()
    test_Point2PlaneDistance()
    test_LUDecomp()
    test_MatrixReduce()
    test_MatrixMergeRow()
    test_PrintMatrix()

run_tests()