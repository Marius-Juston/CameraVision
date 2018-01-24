import vtk
from numpy import random


class VtkPointCloud:

    def __init__(self, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point, color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            self.Colors.InsertNextTuple3(*color[:])
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()
        self.Colors.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName('ColorArray')

        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetScalars(self.Colors)
        self.vtkPolyData.GetPointData().SetActiveScalars('ColorArray')


# TODO make it so that it is a singleton
def start_point_cloud():
    pointCloud = VtkPointCloud()
    renderer = vtk.vtkRenderer()

    # Renderer
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(0, 0, 0)
    renderer.ResetCamera()
    renderWindow = vtk.vtkRenderWindow()

    # Render Window
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()

    # Interactor
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindow.Render()

    # Begin Interaction
    renderWindowInteractor.Start()

    return pointCloud
