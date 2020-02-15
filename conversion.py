import os
import glob
import numpy as np
import vtk

import argparse

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Name of the folder with inputs')
    parser.add_argument('--output', help='Name of the output folder')
    parser.add_argument('--vtk_template', default='', help='vtk template mesh')
    args = parser.parse_args()
    return args

def readVTK(fn):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fn)
    reader.Update()
    poly = cleanPolyData(reader.GetOutput(), 0.)
    coords = vtk_to_numpy(poly.GetPoints().GetData())
    print(coords.shape)
    return coords, poly

def cleanPolyData(poly, tol):
    """
    Cleans a VTK PolyData

    Args:
        poly: VTK PolyData
        tol: tolerance to merge points
    Returns:
        poly: cleaned VTK PolyData
    """

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.SetTolerance(tol)
    clean.PointMergingOn()
    clean.Update()

    poly = clean.GetOutput()
    return poly

def writeVTK(fn, pts, poly):
    poly.GetPoints().SetData(numpy_to_vtk(pts))
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fn)
    writer.SetInputData(poly)
    writer.Update()
    writer.Write()

if __name__ == '__main__':
    args = parse()
    if args.vtk_template != '':
        _, poly = readVTK(args.vtk_template)
        fns = glob.glob(os.path.join(args.input, '*.npy'))
        for fn in fns:
            print(fn)
            out_fn = os.path.join(args.output, os.path.splitext(os.path.basename(fn))[0] + '.vtp')
            writeVTK(out_fn, np.load(fn),poly)
    else:
        fns = glob.glob(os.path.join(args.input, '*.vtp'))
        for fn in fns:
            print(fn)
            coords, poly = readVTK(fn)
            out_fn = os.path.join(args.output, os.path.splitext(os.path.basename(fn))[0] + '.npy')
            np.save(out_fn, coords)


