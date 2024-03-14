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
    AssertWarn(len(aVec) == len(bVec), "palmConfig.VecMax()")
        
    lenMin = min(len(aVec), len(bVec))
    return [max(aVec[ii], bVec[ii]) for ii in range(lenMin)]

def VecMin(aVec, bVec):
    AssertWarn(len(aVec) == len(bVec), "palmConfig.VecMax()")
        
    lenMin = min(len(aVec), len(bVec))
    return [min(aVec[ii], bVec[ii]) for ii in range(lenMin)]

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
    for ii in listX:
        if ii == xx:
            return True
    return False

def Contains(list1, list2):
    for it in list2:
        if not InSet(list1, it):
            return False
    return True

def VectorRange(vv, iMin = FMin, iMax = FMax):
    return max(min(vv), iMin), min(max(vv), iMax)

def VectorCmpAllGt(vv, sc):
    for xx in vv:
        if xx <= sc:
            return False
    return True

def VectorCmpAllLt(vv, sc):
    for xx in vv:
        if xx >= sc:
            return False
    return True

def VectorCmp(vv1, vv2):
    subV = VectorSub(vv1, vv2)
    if VectorCmpAllGt(subV, 0):
        return 1
    if VectorCmpAllLt(subV, 0):
        return -1
    return 0

def VectorAdd(vv1, vv2):
    mLen = min(len(vv1), len(vv2))
    return [vv1[ii] + vv2[ii] for ii in range(mLen)] 

def VectorSub(vv1, vv2):
    mLen = min(len(vv1), len(vv2))
    return [vv1[ii] - vv2[ii] for ii in range(mLen)] 

def VectorMul(vv1, vv2):
    mLen = min(len(vv1), len(vv2))
    return [vv1[ii] * vv2[ii] for ii in range(mLen)] 

def Normalize(vv, iMin, iMax):
    sc = iMax - iMin
    Assert(PositiveNumber(sc), f"palmConfig.Normalize, invalid iMin:{iMin}, iMax:{iMax}")
    return [(vv[ii] - iMin)/sc for ii in range(len(vv))]

def Denormalize(vv, iMin, iMax):
    sc = iMax - iMin
    Assert(PositiveNumber(sc), f"palmConfig.Normalize, invalid iMin:{iMin}, iMax:{iMax}")
    return [vv[ii] * sc + iMin for ii in range(len(vv))]

def VectorSum(vv):
    rtnV = 0
    for nn in vv:
        rtnV += nn
    return rtnV

def VectorDotMul(vv1, vv2):
    return VectorSum(VectorMul(vv1, vv2))

def VectorABS(vv):
    # ABS stands absolute value as vector enclosed by bar '|'.
    # It is the length of the vector
    return math.sqrt(VectorDotMul(vv, vv))

def VectorSMul(vv, scale=1.0):
    return [xx*scale for xx in vv]

def VectorCoss(vv1, vv2, scale=1.0):
    return VectorDotMul(vv1, vv2) / scale

def VectorEuclideanD(vv1, vv2):
    return VectorABS(VectorSub(vv1, vv2))

def VectorManhattanD(vv1, vv2):
    return VectorSum([abs(vv) for vv in VectorSub(vv1, vv2)])

# cosin similarity
def VectorCosins(vv1, vv2):
    return VectorDotMul(vv1, vv2)/VectorABS(vv1)/VectorABS(vv2)

def NormalVector(wm, bb):
    # linear equations defined by wm * X = bb
    # return X to satisfy all the linear equations.
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

    LUL = [[0 for ii in range(rnk)] for ii in range(rnk)]
    LUU = [[0 for ii in range(rnk)] for ii in range(rnk)]

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
