import vtk

GLOBAL_INCREMENT = 1


class InteractorEventHandler(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, renderer, render_window_interactor, render_method, pitch_increment=GLOBAL_INCREMENT,
                 yaw_increment=GLOBAL_INCREMENT,
                 roll_increment=GLOBAL_INCREMENT, zoom_increment=1.01,
                 x_axis_increment=GLOBAL_INCREMENT, y_axis_increment=GLOBAL_INCREMENT,
                 z_axis_increment=GLOBAL_INCREMENT):

        self.camera = vtk.vtkCamera()
        render_window_interactor.SetNumberOfFlyFrames(1)
        self.camera.SetFocalDisk(1000)
        renderer.SetActiveCamera(self.camera)
        renderer.WorldToDisplay()

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

        self.important_keys = {"Up": False, "Down": False, "Right": False, "Left": False, "w": False, "a": False,
                               "s": False, "d": False, "z": False, "x": False, "minus": False, "equal": False,
                               'space': False}

        self.AddObserver("MiddleButtonPressEvent", self.middle_button_press_event)
        self.AddObserver("MiddleButtonReleaseEvent", self.middle_button_release_event)
        self.AddObserver("KeyPressEvent", self.key_press)
        self.AddObserver("KeyReleaseEvent", self.key_release)
        self.AddObserver("MouseEvent", self.key_press)

    def middle_button_press_event(self, obj, event):
        print("Middle Button pressed")
        self.OnMiddleButtonDown()
        return

    def middle_button_release_event(self, obj, event):
        print("Middle Button released")
        self.OnMiddleButtonUp()
        return

    def key_press(self, obj, event):
        key = self.render_window_interactor.GetKeySym()

        if key in self.important_keys.keys():
            camera = self.camera

            x, y, z = camera.GetPosition()
            f_x, f_y, f_z = camera.GetFocalPoint()

            keys = self.important_keys
            keys[key] = True

            if keys["Up"]:
                camera.SetFocalPoint(f_x, f_y + self.y_axis_increment, f_z)

            elif keys["Down"]:
                camera.SetFocalPoint(f_x, f_y - self.y_axis_increment, f_z)

            elif keys["Right"]:
                camera.SetFocalPoint(f_x + self.x_axis_increment, f_y, f_z)

            elif keys["Left"]:
                camera.SetFocalPoint(f_x - self.x_axis_increment, f_y, f_z)

            elif keys["minus"]:
                camera.Zoom(1 / self.zoom_increment)

            elif keys["equal"]:
                camera.Zoom(self.zoom_increment)

            # Rotation is caused because of focus point not changing position as well so the camera  is always point at
            # that position
            elif keys["z"]:
                camera.SetPosition(x, y + self.y_axis_increment, z)
                camera.SetFocalPoint(f_x, f_y + self.y_axis_increment, f_z)

            elif keys["x"]:
                camera.SetPosition(x, y - self.y_axis_increment, z)
                camera.SetFocalPoint(f_x, f_y - self.y_axis_increment, f_z)

            elif keys["d"]:
                camera.SetPosition(x + self.x_axis_increment, y, z)
                camera.SetFocalPoint(f_x + self.x_axis_increment, f_y, f_z)

            elif keys["a"]:
                camera.SetPosition(x - self.x_axis_increment, y, z)
                camera.SetFocalPoint(f_x - self.x_axis_increment, f_y, f_z)

            elif keys["space"]:  # 'r' also resets position but maybe more cleanly
                camera.SetPosition(0, 0, 1)
                camera.SetFocalPoint(0, 0, 0)

            elif keys['s']:
                camera.SetPosition(x, y, z + self.z_axis_increment)
                camera.SetFocalPoint(f_x, f_y, f_z + self.z_axis_increment)

            elif keys['w']:
                camera.SetPosition(x, y, z - self.z_axis_increment)
                camera.SetFocalPoint(f_x, f_y, f_z - self.z_axis_increment)

            self.render()
        self.renderer.ResetCameraClippingRange()

        # print(key, (x, y, z), (pitch, yaw, roll), (f_x, f_x, f_z))
        return

    def key_release(self, obj, event):
        key = self.render_window_interactor.GetKeySym()
        if key in self.important_keys.keys():
            self.important_keys[key] = False
