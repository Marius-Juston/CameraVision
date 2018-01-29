# coding=utf-8
import vtk


# TODO decide if this object should be made private or not
class VtkPointCloud(object):
    """

    """

    def __init__(self, min_value=-10, max_value=10, max_num_points=1e6, window_background_color=(1, 1, 1),
                 full_screen=False, pitch_increment=.5, yaw_increment=.5, roll_increment=.5, zoom_increment=1.01,
                 x_axis_increment=.5, y_axis_increment=.5, z_axis_increment=.5):
        self.y_axis_increment = y_axis_increment
        self.x_axis_increment = x_axis_increment
        self.z_axis_increment = z_axis_increment
        self.zoom_increment = zoom_increment
        self.roll_increment = roll_increment
        self.yaw_increment = yaw_increment
        self.pitch_increment = pitch_increment
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

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.vtkActor)
        self.renderer.SetBackground(
            window_background_color)  # color of bacground (I believe the color range is from [0,1])

        self.renderer.ResetCamera()
        self.render_window = vtk.vtkRenderWindow()

        # Render Window
        self.render_window.AddRenderer(self.renderer)
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()

        self.__setup_keyboard_input(self.render_window_interactor)

        # Interactor
        self.render_window_interactor.SetInteractorStyle(self.camera_event_handler)
        self.render_window_interactor.SetRenderWindow(self.render_window)

        if full_screen:
            self.render_window.FullScreenOn()

        self.render_window.Render()

    def __setup_keyboard_input(self, render_window_interactor):
        self.camera_event_handler = vtk.vtkInteractorStyleTrackballCamera()

        def KeyPress(obj, event):
            camera = self.renderer.GetActiveCamera()
            pitch, yaw, roll = camera.GetOrientation()
            key = render_window_interactor.GetKeySym()

            x, y, z = camera.GetPosition()

            # f_x, f_y, f_z = camera.GetFocalPoint()

            if key == "Up":
                camera.Pitch(-self.pitch_increment)

            elif key == "Down":
                camera.Pitch(self.pitch_increment)

            elif key == "Right":
                camera.Yaw(self.yaw_increment)

            elif key == "Left":
                camera.Yaw(-self.yaw_increment)

            elif key == "minus":
                camera.Zoom(1 / self.zoom_increment)

            elif key == "equal":
                camera.Zoom(self.zoom_increment)

            elif "w":
                # TODO make W make the camera go forwards depending on the orientation of the camera always trying to go the focal point
                pass
                #                  x y  z
                # camera.SetPosition(0,self.y_axis_increment, 1)
            elif "space":
                self.reset_camera()

            self.render()

            print(key, (x, y, z), (pitch, yaw, roll))
            return

        self.camera_event_handler.AddObserver("KeyPressEvent", KeyPress)

    def add_point(self, point, color, render=True):
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

            if render:
                self.render()

    def add_points(self, points, colors):
        """

        :param points:
        :param colors:
        """
        for point, color in zip(points, colors):
            self.add_point(point, color, False)

        self.render()

    def render(self):
        self.render_window.Render()

    def clear_points(self):
        """

        """
        self.__vtkPoints = vtk.vtkPoints()
        self.__vtkCells = vtk.vtkCellArray()
        self.__vtkColors = vtk.vtkUnsignedCharArray()
        self.__vtkColors.SetNumberOfComponents(3)
        self.__vtkColors.SetName('ColorArray')

        self.__vtkPolyData.SetPoints(self.__vtkPoints)
        self.__vtkPolyData.SetVerts(self.__vtkCells)
        self.__vtkPolyData.GetPointData().SetScalars(self.__vtkColors)

    def close(self):
        """

        """
        del self.__vtkPoints
        del self.__vtkCells
        del self.__vtkColors
        del self.__vtkPolyData
        del self.vtkActor
        del self.renderer
        del self.render_window

        self.render_window_interactor.TerminateApp()

        del self.render_window_interactor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.start()
        self.close()

    def start(self):
        self.render_window_interactor.Start()

    def reset_camera(self, camera=None):
        if camera is None:
            camera = self.renderer.GetActiveCamera()
