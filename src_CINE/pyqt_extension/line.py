import numpy as np
from vispy import app, scene, color, visuals
from vispy import app, scene
from PyQt6.QtCore import pyqtSignal, QObject


class EditLineVisual(scene.visuals.InfiniteLine):
    def __init__(self, *args, **kwargs):
        scene.visuals.InfiniteLine.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.last_position = np.inf
        self.freeze()

    def on_draw(self, event):
        scene.visuals.InfiniteLine.draw(self)

    def on_mouse_move(self, pos_scene):
        if (abs(pos_scene - self.last_position) >= 1):
            self.set_data(pos=round(pos_scene), color=[1, 1, 1, 0.5])
            self.last_position = round(pos_scene)
            self.update()


class Edited_Canvas(scene.SceneCanvas, QObject):
    """ A simple test canvas for testing the EditLineVisual """
    vertical_line_moved_signal = pyqtSignal(int)
    horizon_line_moved_signal = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        QObject.__init__(self)
        scene.SceneCanvas.__init__(self, *args, **kwargs)

        self.unfreeze()
        self.vertical_line_position = 0
        self.horizon_line_position = 0
        self.view = self.central_widget.add_view()

        self.vertical_line_limit = [0, 180]
        self.horizon_line_limit = [0, 180]

        self.vertical_line = None
        self.horizon_line = None

        self.is_pressed = False

        self.freeze()
        self.set_view(self.view)
        self.show()

    def set_view(self, view):
        self.central_widget.remove_widget(self.view)
        self.view = view
        self.central_widget.add_widget(self.view)
        self.vertical_line = EditLineVisual(pos=self.vertical_line_position, color=[1.0, 1, 0.5, 0.5], vertical=True,
                                            parent=self.view.scene)
        self.vertical_line.transform = visuals.transforms.STTransform(translate=(0, 0, -10))
        self.horizon_line = EditLineVisual(pos=self.horizon_line_position, color=[1.0, 1.0, 0.5, 0.5], vertical=False,
                                           parent=self.view.scene)
        self.horizon_line.transform = visuals.transforms.STTransform(translate=(0, 0, -10))

    def set_camera(self, camera):
        self.view.camera = camera
        self.view.camera._viewbox.events.mouse_move.disconnect(
            self.view.camera.viewbox_mouse_event)

    def set_limit(self, vertical_left, vertical_right, horizon_down, horizon_up):
        self.vertical_line_limit = [vertical_left, vertical_right]
        self.horizon_line_limit = [horizon_down, horizon_up]

    def get_vertical_line_position(self):
        return self.vertical_line_position

    def get_horizon_line_position(self):
        return self.horizon_line_position

    def set_vertical_pos(self, pos):
        pos = self.get_vertical_pos(pos)
        self.vertical_line.on_mouse_move(pos)
        self.vertical_line_position = self.vertical_line.last_position

    def set_horizon_pos(self, pos):
        pos = self.get_horizon_pos(pos)
        self.horizon_line.on_mouse_move(pos)
        self.horizon_line_position = self.horizon_line.last_position

    def get_vertical_pos(self, pos):
        if (pos >= self.vertical_line_limit[1]):
            return self.vertical_line_limit[1]
        if (pos <= self.vertical_line_limit[0]):
            return self.horizon_line_limit[0]
        return pos

    def get_horizon_pos(self, pos):
        if (pos >= self.horizon_line_limit[1]):
            return self.horizon_line_limit[1]
        if (pos <= self.horizon_line_limit[0]):
            return self.horizon_line_limit[0]
        return pos

    def on_mouse_press(self, event):
        self.is_pressed = True

    def on_mouse_move(self, event):
        tr = self.scene.node_transform(self.vertical_line)
        pos = tr.map(event.pos)
        if (abs(pos[0] - self.vertical_line_position) < 5 and self.is_pressed):
            self.vertical_line.on_mouse_move(self.get_vertical_pos(pos[0]))
            self.vertical_line_position = self.vertical_line.last_position
            self.vertical_line_moved_signal.emit(self.vertical_line.last_position)
            print(self.vertical_line_position)
        elif (abs(pos[1] - self.horizon_line_position) < 5 and self.is_pressed):
            self.horizon_line.on_mouse_move(self.get_horizon_pos(pos[1]))
            self.horizon_line_position = self.horizon_line.last_position
            self.horizon_line_moved_signal.emit(self.horizon_line.last_position)
            print(self.horizon_line_position)

    def on_mouse_release(self, event):
        self.vertical_line.set_data(pos=self.vertical_line_position, color=[1.0, 1, 0.5, 0.5])
        self.horizon_line.set_data(pos=self.horizon_line_position, color=[1.0, 1.0, 0.5, 0.5])
        self.vertical_line.update()
        self.horizon_line.update()
        self.is_pressed = False

    def reset(self):
        self.vertical_line_position = 0
        self.horizon_line_position = 0
        self.vertical_line = None
        self.horizon_line = None

        view = scene.widgets.ViewBox(parent=self.scene)
        self.set_view(view)


if __name__ == '__main__':
    win = Edited_Canvas()
    app.run()
