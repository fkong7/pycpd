from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import deformable_registration
import numpy as np
import time
#import vtk

#def visualize(iteration, error, X, Y, ax):
#    plt.cla()
#    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
#    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
#    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
#    ax.legend(loc='upper left', fontsize='x-large')
#    plt.draw()
#    plt.pause(0.001)

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
    fn_X = ''
    fn_Y = ''
    fn_Y_out = ''
    X = np.load(fn_X)
    Y = np.load(fn_Y)
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    reg = deformable_registration(**{ 'X': X-X_mean, 'Y': Y-Y_mean})
    TY, _ = reg.register()
    TY += Y_mean
    np.save(fn_Y_out, TY)
    #writePD('/Users/fanweikong/Downloads/phase9_6.vtk', TY, poly)


if __name__ == '__main__':
    main()
