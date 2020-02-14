from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import deformable_registration
import numpy as np
import time
import vtk

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def readPD(fn):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fn)
    reader.Update()
    poly = reader.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    coords -= np.mean(coords, axis=0)
    return coords, poly
def writePD(fn, pts, poly):
    from vtk.util.numpy_support import numpy_to_vtk
    poly.GetPoints().SetData(numpy_to_vtk(pts))
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fn)
    writer.SetInputData(poly)
    writer.Update()
    writer.Write()

def main():
    X, _ = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase6_deci.nii.vtk')

    Y, poly  = readPD('/Users/fanweikong/Documents/Modeling/SurfaceModeling/Label_based/MACS40282_20150504/surfaces/phase9_deci.nii.vtk')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    reg = deformable_registration(**{ 'X': X, 'Y': Y })
    TY, _ = reg.register()
    
    writePD('/Users/fanweikong/Downloads/phase9_6.vtk', TY, poly)


if __name__ == '__main__':
    main()
