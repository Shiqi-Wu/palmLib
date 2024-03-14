import sys
from enum import Enum
import cupy as cp
import math

class InfoLevel(Enum):
    TRACE = 0
    DEBUG = 1
    ERROR = 2
    WARN  = 3
    INFO  = 4

def Assert(cnd, msg):
    if not cnd:
        Print(InfoLevel.ERROR, f"Assert Fail:{msg}")
    return cnd

def AssertWarn(cnd, msg):
    if not cnd:
        Print(InfoLevel.WARN, f"Assert Warn:{msg}")
    return cnd

def VecMax(aVec, bVec):
    aVec = cp.array(aVec)
    bVec = cp.array(bVec)
    AssertWarn(aVec.shape == bVec.shape, "palmConfig.VecMax()")
    return cp.maximum(aVec, bVec)

def VecMin(aVec, bVec):
    aVec = cp.array(aVec)
    bVec = cp.array(bVec)
    AssertWarn(aVec.shape == bVec.shape, "palmConfig.VecMin()")
    return cp.minimum(aVec, bVec)

def Print(iLevel, str):
    if iLevel.value >= RunningLevel.value:
        print(str)


RunningLevel = InfoLevel.TRACE
FMax = sys.float_info.max
FMin = - sys.float_info.max
Fmin = sys.float_info.min

def SetRunningLevel(iLevel):
    RunningLevel = iLevel

def PositiveNumber(nn):
    return nn >= Fmin

def InSet(listX, xx):
    listX = cp.array(listX)
    return cp.any(listX == xx)

def Contains(list1, list2):
    list1 = cp.array(list1)
    list2 = cp.array(list2)
    return cp.all(cp.isin(list2, list1))

def VectorRange(vv, iMin=FMin, iMax=FMax):
    vv = cp.array(vv)
    return max(cp.min(vv), iMin), min(cp.max(vv), iMax)

def VectorCmpAllGt(vv, sc):
    vv = cp.array(vv)
    return cp.all(vv > sc)

def VectorCmpAllLt(vv, sc):
    vv = cp.array(vv)
    return cp.all(vv < sc)

def VectorCmp(vv1, vv2):
    vv1 = cp.array(vv1)
    vv2 = cp.array(vv2)
    subV = VectorSub(vv1, vv2)
    if VectorCmpAllGt(subV, 0):
        return 1
    if VectorCmpAllLt(subV, 0):
        return -1
    return 0

def VectorAdd(vv1, vv2):
    vv1 = cp.array(vv1)
    vv2 = cp.array(vv2)
    mLen = min(vv1.size, vv2.size)
    return vv1[:mLen] + vv2[:mLen]

def VectorSub(vv1, vv2):
    vv1 = cp.array(vv1)
    vv2 = cp.array(vv2)
    mLen = min(vv1.size, vv2.size)
    return vv1[:mLen] - vv2[:mLen]

def VectorMul(vv1, vv2):
    vv1, vv2 = cp.array(vv1), cp.array(vv2)
    mLen = min(vv1.size, vv2.size)
    return vv1[:mLen] * vv2[:mLen]

def Normalize(vv, iMin, iMax):
    vv = cp.array(vv)
    sc = iMax - iMin
    Assert(PositiveNumber(sc), "Invalid iMin/iMax")
    return (vv - iMin) / sc

def Denormalize(vv, iMin, iMax):
    vv = cp.array(vv)
    sc = iMax - iMin
    Assert(PositiveNumber(sc), "Invalid iMin/iMax")
    return vv * sc + iMin

def VectorSum(vv):
    vv = cp.array(vv)
    return cp.sum(vv)

def VectorDotMul(vv1, vv2):
    vv1, vv2 = cp.array(vv1), cp.array(vv2)
    return VectorSum(VectorMul(vv1, vv2))

def VectorABS(vv):
    vv = cp.array(vv)
    return cp.sqrt(VectorDotMul(vv, vv))

def VectorSMul(vv, scale=1.0):
    vv = cp.array(vv)
    return vv * scale

def VectorCoss(vv1, vv2, scale=1.0):
    vv1, vv2 = cp.array(vv1), cp.array(vv2)
    return VectorDotMul(vv1, vv2) / scale

def VectorEuclideanD(vv1, vv2):
    vv1, vv2 = cp.array(vv1), cp.array(vv2)
    return VectorABS(VectorSub(vv1, vv2))

def VectorManhattanD(vv1, vv2):
    vv1, vv2 = cp.array(vv1), cp.array(vv2)
    return VectorSum(cp.abs(VectorSub(vv1, vv2)))

# cosin similarity
def VectorCosins(vv1, vv2):
    return VectorDotMul(vv1, vv2)/VectorABS(vv1)/VectorABS(vv2)

def NormalVector(wm, bb):
    # linear equations defined by wm * X = bb
    # return X to satisfy all the linear equations.
    wm = cp.array(wm)
    bb = cp.array(bb)
    rtnV = None
    try:
        rtnV = cp.linalg.solve(wm, bb)
    except cp.linalg.LinAlgError as e:
        Print(InfoLevel.ERROR, f"NormalVector(LinAlgError): {e}")
    return rtnV

def Point2PlaneDistance(pt, nv, bb):
    # Plane is defined by nv * X = bb
    return (VectorDotMul(pt, nv) - bb) / VectorABS(nv)

def LUDecomp(mat):
    rnk = len(mat)
    mat = cp.array(mat)

    LUL = cp.zeros((rnk, rnk))
    LUU = cp.zeros((rnk, rnk))

    for ii in range(rnk):
        for kk in range(ii, rnk):
            ss = 0
            for jj in range(ii):
                ss += LUL[ii][jj] * LUU[jj][kk]
            LUU[ii][kk] = mat[ii][kk] - ss
          #  print(f"LUU[{ii},{kk}] = {LUU[ii][kk]}")

        if abs(LUU[ii][ii]) <= Fmin:
            print(f"LUU[{ii}] = {LUU[ii][ii]}, too small")
            LUL = None # mark the incomplete of LU decompostion
            break

        for kk in range(ii, rnk):
            if ii == kk:
                LUL[ii][ii] = 1
            else:
                ss = 0
                for jj in range(ii):
                    ss += LUL[kk][jj] * LUU[jj][ii]

                LUL[kk][ii] = (mat[kk][ii] - ss) / LUU[ii][ii] 

          #      print(f"LUL[{kk},{ii}] = {LUL[kk][ii]}")

    return LUL, LUU

def MatrixReduce(mat, rowName, colName):
    return mat, rowName, colName

def MatrixMergeRow(mat, rowName, colName, mergeIdxList):
    return mat, rowName, colName

def PrintMatrix(mat):
    for vv in mat:
        print(vv)
