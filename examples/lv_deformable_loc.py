from functools import partial
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../pycpd'))
import glob
from deformable_registration import deformable_registration
from rigid_registration import rigid_registration
from affine_registration import affine_registration
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def readPD(fn):
    import vtk
    if os.path.splitext(fn)[-1] == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fn)
    reader.Update()
    poly = reader.GetOutput()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(0.)
    clean.PointMergingOn()
    clean.Update()

    poly = clean.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    mean = np.mean(coords, axis=0)
    coords -= mean
    print(coords.shape)
    return coords, poly, mean 

def writePD(fn, pts, poly):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    poly.GetPoints().SetData(numpy_to_vtk(pts))
    if os.path.splitext(fn)[-1] =='.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fn)
    writer.SetInputData(poly)
    writer.Update()
    writer.Write()

def main():
    X, _, mean_X = readPD('/Users/fanweikong/Documents/Modeling/pycpd/data/test/TRV4P8CTAI_1.nii.gz.vtk')
    Y, poly, mean_Y  = readPD('/Users/fanweikong/Documents/Modeling/pycpd/data/test/TRV4P1CTAI_1.nii.gz.vtk')
    #print(X)
    #print(Y)
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #callback = partial(visualize, ax=ax)
    #
    #reg = deformable_registration(**{ 'X': X, 'Y': Y})
    #reg.register(callback)
    #plt.show()
    ##TY += mean_X
    #X, _, mean_X = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase6_deci.nii.vtk')

    #Y, poly, mean_Y  = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase9_deci.nii.vtk')
    print(X)
    print(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    reg = affine_registration(**{ 'X': X, 'Y': Y})
    Y, _ = reg.register(rank=1)
    reg = deformable_registration(**{ 'X': X, 'Y': Y})
    Y, _ = reg.register(rank=1)
    reg = deformable_registration(tolerance=1e-4,alpha=0.005, beta=1000.,**{ 'X': X, 'Y': Y })
    Y, _ = reg.register(rank=1)
    reg = deformable_registration(tolerance=1e-4,alpha=0.005, beta=1000.,**{ 'X': X, 'Y': Y })
    TY, _ = reg.register(rank=1)
    
    TY += mean_X
    #writePD('/Users/fanweikong/Downloads/phase9_6.vtk', TY, poly)
    writePD('/Users/fanweikong/Documents/Modeling/pycpd/data/registered/test.vtp', TY, poly)

if __name__ == '__main__':
    main()
