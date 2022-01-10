<h1>Medical Image CINE Viewer</h1>
An application that displays 4D (3d + time) cardiac CT images and quickly shows statistical reports for analyzing heart movement over the period of time

## Built with
- [Vispy](https://github.com/vispy/vispy): high-performance interactive 2D/3D data visualization library
- [Pyqt](https://www.riverbankcomputing.com/static/Docs/PyQt6): widget toolkit for creating graphical user interfaces

## Features

- display images in 4D (3D + time) and 3 different planes: the axial, sagittal, and coronal planes
- Works with Nifti, Tiff, and CT images
- Drag/Drop features for loading volume or label images
- Generate statistical reports (size, and correlation of variation)

## Getting Started

1. Clone the repo

```shell
git clone https://github.com/loctran15/Medical_Image_CINE_Viewer.git
```

2.  Change the current directory to the repo folder

```shell
cd [Medical_Image_CINE_Viewer]
```

5. Install python packages

```shell
pip install -r requirements.txt
```

6. Run the app in the development mode.

```shell
python CINE_main.py 
```

## License

MIT license. See `LICENSE` for more information.
