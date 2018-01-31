import vtk


class InteractorEventHandler(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, render_window_interactor, render_method, pitch_increment=.5, yaw_increment=.5,
                 roll_increment=.5, zoom_increment=1.01,
                 x_axis_increment=.5, y_axis_increment=.5, z_axis_increment=.5):
        self.render = render_method
        self.zoom_increment = zoom_increment
        self.roll_increment = roll_increment
        self.x_axis_increment = x_axis_increment
        self.y_axis_increment = y_axis_increment
        self.yaw_increment = yaw_increment
        self.pitch_increment = pitch_increment
        self.render_window_interactor = render_window_interactor
        self.renderer = renderer
        self.z_axis_increment = z_axis_increment
        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("KeyPressEvent", self.key_press)

    def middle_button_press_event(self, obj, event):
        print("Middle Button pressed")
        self.OnMiddleButtonDown()
        return

    def middle_button_release_event(self, obj, event):
        print("Middle Button released")
        self.OnMiddleButtonUp()
        return

    def key_press(self, obj, event):
        print(
            obj
        )

        camera = self.renderer.GetActiveCamera()
        pitch, yaw, roll = camera.GetOrientation()
        key = self.render_window_interactor.GetKeySym()

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
