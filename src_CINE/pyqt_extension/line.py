import numpy as np
from vispy import app, scene, color, visuals
from vispy import app, scene
from PyQt6.QtCore import pyqtSignal, QObject


class Edited_Canvas(scene.SceneCanvas, QObject):

    def __init__(self, *args, **kwargs):
        QObject.__init__(self)
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.freeze()
        self.set_view(self.view)
        self.show()

    def set_view(self, view):
        self.central_widget.remove_widget(self.view)
        self.view = view
        self.central_widget.add_widget(self.view)

    def set_camera(self, camera):
        self.view.camera = camera
        self.view.camera._viewbox.events.mouse_move.disconnect(
            self.view.camera.viewbox_mouse_event)

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
