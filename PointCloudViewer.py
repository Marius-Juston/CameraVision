import vtk


# TODO decide if this object should be made private or not
class VtkPointCloud:

    def __init__(self, maxNumPoints=1e6):
        self.__maxNumPoints = maxNumPoints
        self.__vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.__vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point, color):
        if self.__vtkPoints.GetNumberOfPoints() < self.__maxNumPoints:
            pointId = self.__vtkPoints.InsertNextPoint(point[:])
            self.__vtkDepth.InsertNextValue(point[2])
            self.__vtkCells.InsertNextCell(1)
            self.__vtkCells.InsertCellPoint(pointId)
            self.__vtkColors.InsertNextTuple3(*color)
            self.__vtkCells.Modified()
            self.__vtkPoints.Modified()
            self.__vtkDepth.Modified()
            self.__vtkColors.Modified()

    def clearPoints(self):
        del self.__vtkPoints
        del self.__vtkCells
        del self.__vtkDepth
        del self.__vtkColors

        self.__vtkPoints = vtk.vtkPoints()
        self.__vtkCells = vtk.vtkCellArray()
        self.__vtkDepth = vtk.vtkDoubleArray()
        self.__vtkDepth.SetName('DepthArray')
        self.__vtkColors = vtk.vtkUnsignedCharArray()
        self.__vtkColors.SetNumberOfComponents(3)
        self.__vtkColors.SetName('ColorArray')

        self.__vtkPolyData.SetPoints(self.__vtkPoints)
        self.__vtkPolyData.SetVerts(self.__vtkCells)
        self.__vtkPolyData.GetPointData().SetScalars(self.__vtkDepth)
        self.__vtkPolyData.GetPointData().SetScalars(self.__vtkColors)
        self.__vtkPolyData.GetPointData().SetActiveScalars('ColorArray')


# TODO make it so that it is a singleton
def start_point_cloud():
    pointCloud = VtkPointCloud()

    renderer = vtk.vtkRenderer()

    # Renderer
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(0, 0, 0)  # color of bacground (I believe the color range is from [0,1])
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
