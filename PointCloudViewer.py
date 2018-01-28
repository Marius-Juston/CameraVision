# coding=utf-8
import vtk


# TODO decide if this object should be made private or not
class VtkPointCloud(object):
    """

    """

    def __init__(self, min_value=-10, max_value=10, max_num_points=1e6):
        self.__maxNumPoints = max_num_points
        self.__vtkPolyData = vtk.vtkPolyData()
        self.clear_points()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.__vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarVisibility(1)
        mapper.SetScalarRange(min_value, max_value)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.GetProperty().SetPointSize(5)
        # self.vtkActor.DragableOn()

        self.vtkActor.SetMapper(mapper)

    def add_point(self, point, color):
        """

        :param point:
        :param color:
        """
        if self.__vtkPoints.GetNumberOfPoints() < self.__maxNumPoints:
            point_id = self.__vtkPoints.InsertNextPoint(point[:])
            self.__vtkCells.InsertNextCell(1)
            self.__vtkCells.InsertCellPoint(point_id)
            self.__vtkColors.InsertNextTuple3(*color)
        self.__vtkCells.Modified()
        self.__vtkPoints.Modified()
        self.__vtkColors.Modified()

    def add_points(self, points, colors):
        """

        :param points:
        :param colors:
        """
        for point, color in zip(points, colors):
            self.add_point(point, color)

    def clear_points(self):
        """

        """
        # del self.__vtkPoints
        # del self.__vtkCells
        # del self.__vtkDepth
        # del self.__vtkColors

        self.__vtkPoints = vtk.vtkPoints()
        self.__vtkCells = vtk.vtkCellArray()
        self.__vtkColors = vtk.vtkUnsignedCharArray()
        self.__vtkColors.SetNumberOfComponents(3)
        self.__vtkColors.SetName('ColorArray')

        self.__vtkPolyData.SetPoints(self.__vtkPoints)
        self.__vtkPolyData.SetVerts(self.__vtkCells)
        self.__vtkPolyData.GetPointData().SetScalars(self.__vtkColors)
        # self.__vtkPolyData.GetPointData().SetActiveScalars('ColorArray')

    def close(self):
        """

        """
        del self.__vtkPoints
        del self.__vtkCells
        del self.__vtkColors
        del self.__vtkPolyData
        del self.vtkActor
        del self.vtkActor
        del self.vtkActor
        del self.vtkActor
        del self.vtkActor


# TODO make it so that it is a singleton
def start_point_cloud():
    """

    :return:
    """
    point_cloud = VtkPointCloud()

    renderer = vtk.vtkRenderer()

    # Renderer
    renderer.AddActor(point_cloud.vtkActor)
    renderer.SetBackground(1, 1, 1)  # color of bacground (I believe the color range is from [0,1])
    renderer.ResetCamera()
    render_window = vtk.vtkRenderWindow()

    # Render Window
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()

    # Interactor
    render_window_interactor.SetRenderWindow(render_window)
    render_window.Render()

    # renderWindowInteractor.CreateRepeatingTimer(0)

    # Begin Interaction
    # renderWindowInteractor.Start()

    return point_cloud, render_window, render_window_interactor
