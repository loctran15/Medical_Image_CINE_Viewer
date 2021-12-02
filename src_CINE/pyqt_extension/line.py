import numpy as np
from vispy import app, scene, color, visuals
from vispy import app, scene
from PyQt6.QtCore import pyqtSignal, QObject
from typing import Any


class Edited_Canvas(scene.SceneCanvas, QObject):
    mouse_pressed_signal = pyqtSignal(str, int, int)

    def on_mouse_press(self, event):
        # 1=left, 2=right , 3=middle button
        if event.button == 1:
            self.mouse_pressed = True
            self.get_scene_coord(event.pos)

    def on_mouse_move(self, event):
        if (self.mouse_pressed):
            self.get_scene_coord(event.pos)

    def get_scene_coord(self, event_pos):
        # update the vertical and horizontal line
        tr = self.scene.node_transform(self.view.scene)
        pos = tr.map(event_pos)
        pos_x = int(pos[0])
        pos_y = int(pos[1])
        if (isinstance(self.visual_at([pos_x, pos_y]), scene.widgets.viewbox.ViewBox)):
            # emit the signal
            self.mouse_pressed_signal.emit(str(id(self)), pos_x, pos_y)

    def on_mouse_release(self, event):
        self.mouse_pressed = False

    def __init__(self, *args, **kwargs):
        QObject.__init__(self)
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.mouse_pressed = False
        self.view = self.central_widget.add_view()
        self.vertical_line: visuals.InfiniteLineVisual = None
        self.horizontal_line: visuals.InfiniteLineVisual = None
        self.freeze()
        self.set_view(self.view)
        self.show()

    def set_view(self, view):
        self.central_widget.remove_widget(self.view)
        self.view = view
        self.central_widget.add_widget(self.view)
        self.vertical_line = scene.InfiniteLine(pos=50, color=[1.0, 1, 0.5, 0.5], vertical=True, line_width=0.5,
                                                parent=self.view.scene)
        self.vertical_line.transform = visuals.transforms.STTransform(translate=(0, 0, -1))
        self.vertical_line.set_gl_state(depth_test=True)
        self.horizontal_line = scene.InfiniteLine(pos=50, color=[1.0, 1, 0.5, 0.5], vertical=False, line_width=0.5,
                                                  parent=self.view.scene)
        self.horizontal_line.transform = visuals.transforms.STTransform(translate=(0, 0, -1))
        self.horizontal_line.set_gl_state(depth_test=True)

    def set_camera(self, camera):
        self.view.camera = camera
        self.view.camera._viewbox.events.mouse_move.disconnect(
            self.view.camera.viewbox_mouse_event)

    def reset(self):
        self.vertical_line = None
        self.horizontal_line = None

        view = scene.widgets.ViewBox(parent=self.scene)
        self.set_view(view)


if __name__ == '__main__':
    win = Edited_Canvas()
    app.run()
