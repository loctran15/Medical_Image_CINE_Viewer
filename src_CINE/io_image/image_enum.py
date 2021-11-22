from enum import Enum, auto


# enum
class ImageType(Enum):
    GRAYSCALE = auto()
    LABEL = auto()
    GRAYSCALE_LABEL = auto()

    # without this == may return unexpected result
    def __eq__(self, other):
        return self.value == other.value


class ImageFileType(Enum):
    NIFTY = auto()
    DICOM = auto()
    ROI = auto()
    TIFF = auto()

    # without this == may return unexpected result
    def __eq__(self, other):
        return self.value == other.value


class ColorType(Enum):
    GRAYSCALE = auto()
    RGB = auto()
    RGBA = auto()

    # without this == may return unexpected result
    def __eq__(self, other):
        return self.value == other.value


label_dict = {
    "BOX": 1,
    "WH": 2,
    "LUNG": 3,
    "LVM": 5,
    "LV": 6,
    "AO": 9,
    "LIVER": 10,
    "DAS": 11,
    "RV": 12,
    "CW": 13,
    "PV": 14,
    "LA": 15,
    "LAA": 16,
    "RA": 19,
    "IVC": 20,
    "PA": 21,
    "SVC": 22,
    "SPINE": 23,
}

RGB_color_dict = {
    "AO": (1, 1, 0),
    "LV": (1, 0, 1),
    "LVM": (0, 0.5, 0),
    "LA": (1, 0.4, 0.4),
    "RV": (0, 1, 1),
    "RA": (0.4, 0.75, 0.4),
    "LAA": (0.75, 0.4, 0.4),
    "SVC": (0.4, 0.4, 0.75),
    "IVC": (0.4, 0.5, 0.4),
    "PA": (0.4, 0.4, 1),
    "PV": (0, 0.5, 0.5),
    "WH": (0, 1, 0)
}

if __name__ == "__main__":
    print("heelo")
    print("hi")
