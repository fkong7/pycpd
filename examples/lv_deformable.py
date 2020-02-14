from functools import partial
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../pycpd'))
import glob
from deformable_registration import deformable_registration
import numpy as np
import time
from mpi4py import MPI

def readPD(fn):
    import vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fn)
    reader.Update()
    poly = reader.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    mean = np.mean(coords, axis=0)
    coords -= mean
    return coords, poly, mean 
def writePD(fn, pts, poly):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    poly.GetPoints().SetData(numpy_to_vtk(pts))
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fn)
    writer.SetInputData(poly)
    writer.Update()
    writer.Write()

def main():
    #X, _, mean_X = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase6_deci.nii.vtk')
    #Y, poly, mean_Y  = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase9_deci.nii.vtk')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input directory name')
    parser.add_argument('--output', help='Output directory name')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total = comm.Get_size()
    
    fns = sorted(glob.glob(os.path.join(args.input, '*.npy')))
    print("Template: ", fns[0])
    num_vol_per_core = int(np.floor(len(fns)/total))
    extra = len(fns) % total
    vol_ids = list(range(rank*num_vol_per_core,(rank+1)*num_vol_per_core))
    if rank < extra:
        vol_ids.append(len(fns)-1-rank)

    for i in vol_ids:
        fn_X = fns[i]
        fn_Y = fns[0]
        fn_X_out = os.path.join(args.output, os.path.basename(fn_X))
        X = np.load(fn_X)
        Y = np.load(fn_Y)
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        reg = deformable_registration(**{ 'X': X-X_mean, 'Y': Y-Y_mean})
        TY, _ = reg.register()
        TY += Y_mean
        np.save(fn_X_out, TY)
        #writePD('/Users/fanweikong/Downloads/phase9_6.vtk', TY, poly)


if __name__ == '__main__':
    main()
