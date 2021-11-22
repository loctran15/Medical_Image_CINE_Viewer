from enum import Enum, auto
from dataclasses import dataclass, field

from typing import Tuple, Union, Dict, Optional, Any, DefaultDict
import numpy as np

from .exporter import ImageExporter
from .image_enum import ImageType, ImageFileType, ColorType, label_dict, RGB_color_dict

from vispy.scene.widgets.viewbox import ViewBox
from vispy.visuals.isosurface import IsosurfaceVisual
from vispy.scene.visuals import Isosurface
from vispy.color.color_array import Color
from vispy import app, scene, color, visuals
from vispy.visuals.filters import ShadingFilter


@dataclass
class Image3DScene:
    parent: Optional[ViewBox] = field(default=None)
    scene_dict: dict[str, IsosurfaceVisual] = field(default_factory=lambda: {}, repr=False, init=False)
    is_visual_dict: dict[str, bool] = field(
        default_factory=lambda: {'WH': False, 'AO': False, 'LV': False, 'LA': False, 'RV': False,
                                 'RA': False,
                                 'LAA': False, 'SVC': False, 'IVC': False, 'PA': False,
                                 'PV': False, 'LVM': False})

    def setup_scene(self, image_array: np.ndarray,
                    parent: Optional[ViewBox] = None,
                    is_visual_dict: Optional[dict[str, bool]] = None,
                    alpha: int = 1):
        if (is_visual_dict):
            self.is_visual_dict = is_visual_dict.copy()
        if (parent):
            self.parent = parent

        if (self.parent is None):
            print("cannot set up scene without parent")

        for segment_name, value in label_dict.items():
            if (segment_name in RGB_color_dict.keys()):
                new_segment = np.zeros(image_array.shape, dtype=np.uint16)
                new_segment[np.where(image_array == value)] = 1
                self.scene_dict[segment_name] = scene.visuals.Isosurface(new_segment, level=new_segment.max() / 4.,
                                                                         color=(*RGB_color_dict[segment_name], alpha),
                                                                         shading='smooth',
                                                                         parent=self.parent)
                self.scene_dict[segment_name].visible = self.is_visual_dict[segment_name]
                if (self.is_visual_dict[segment_name]):
                    self.scene_dict[segment_name]._prepare_draw(self.scene_dict[segment_name])
                    self.scene_dict[segment_name].shading_filter.shininess = 500
                    # self.scene_dict[segment_name].shading_filter.specular_coefficient = (0, 0, 1, 0.5)
                    self.scene_dict[segment_name].shading_filter.diffuse_coefficient = 0.9
                    self.scene_dict[segment_name].shading_filter.ambient_coefficient = 0.7
                    self.scene_dict[segment_name].shading_filter.light_dir = (0, 1, 0)

    def update_scene(self, is_visual_dict: Optional[dict[str, bool]] = None,
                     alpha: int = None):
        if (not self.scene_dict):
            print("please setup the scene first using .setup_scene()")
            return

        if (is_visual_dict is None):
            is_visual_dict = self.is_visual_dict.copy()

        for segment_name, value in label_dict.items():
            if (segment_name in RGB_color_dict.keys()):
                if (alpha):
                    self.scene_dict[segment_name].set_data(color=(*RGB_color_dict[segment_name], alpha))

                if (self.is_visual_dict[segment_name] != is_visual_dict[segment_name]):
                    self.scene_dict[segment_name].visible = is_visual_dict[segment_name]
                    if (is_visual_dict[segment_name]):
                        self.scene_dict[segment_name]._prepare_draw(self.scene_dict[segment_name])
                        self.scene_dict[segment_name].shading_filter.shininess = 500
                        # self.scene_dict[segment_name].shading_filter.specular_coefficient = (0, 0, 1, 0.5)
                        self.scene_dict[segment_name].shading_filter.diffuse_coefficient = 0.9
                        self.scene_dict[segment_name].shading_filter.ambient_coefficient = 0.7
                        self.scene_dict[segment_name].shading_filter.light_dir = (0, 1, 0)

                self.scene_dict[segment_name].update()

        self.is_visual_dict = is_visual_dict.copy()
