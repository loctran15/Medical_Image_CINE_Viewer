import pyqtgraph as pg
from pyqtgraph import exporters
from src_CINE.io_image.exporter import CSVExporter, color_map
from src_CINE.io_image.image import Image
from pyqtgraph import mkPen


def plot(images: list[Image]) -> None:
    # set background color
    pg.setConfigOption('background', 'w')

    phase = range(1, len(images) + 1)

    WH = [None for _ in phase]
    AO = [None for _ in phase]
    LV = [None for _ in phase]
    LA = [None for _ in phase]
    RV = [None for _ in phase]
    RA = [None for _ in phase]
    LAA = [None for _ in phase]
    SVC = [None for _ in phase]
    IVC = [None for _ in phase]
    PA = [None for _ in phase]
    PV = [None for _ in phase]
    LVM = [None for _ in phase]

    # define the data
    theTitle = "pyqtgraph plot"
    for image in images:
        WH[image.phase - 1] = image.size.WH
        AO[image.phase - 1] = image.size.AO
        LV[image.phase - 1] = image.size.LV
        LA[image.phase - 1] = image.size.LA
        RV[image.phase - 1] = image.size.RV
        RA[image.phase - 1] = image.size.RA
        LAA[image.phase - 1] = image.size.LAA
        SVC[image.phase - 1] = image.size.SVC
        IVC[image.phase - 1] = image.size.IVC
        PA[image.phase - 1] = image.size.PA
        PV[image.phase - 1] = image.size.PV
        LVM[image.phase - 1] = image.size.LVM

    # create plot
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties
    plt.setLabel('left', 'Value', units='V')
    plt.setLabel('bottom', 'Time', units='s')
    # plt.setXRange(0, 10)
    # plt.setYRange(0, 20)
    plt.setWindowTitle('pyqtgraph plot')

    plt.plotItem.setAutoVisible(y=True)
    plt.setAutoVisible(y=True)

    # plot
    plt.plot(phase, WH, pen=mkPen(tuple(color_map["WH"]), width=3), symbol='o', symbolPen=tuple(color_map["WH"]),
             symbolBrush=0.2, name='WH')
    plt.plot(phase, AO, pen=mkPen(tuple(color_map["AO"]), width=3), symbol='o', symbolPen=tuple(color_map["AO"]),
             symbolBrush=0.2,
             name='AO')
    plt.plot(phase, LV, pen=mkPen(tuple(color_map["LV"]), width=3), symbol='o', symbolPen=tuple(color_map["LV"]),
             symbolBrush=0.2,
             name='LV')
    plt.plot(phase, LA, pen=mkPen(tuple(color_map["LA"]), width=3), symbol='o', symbolPen=tuple(color_map["LA"]),
             symbolBrush=0.2,
             name='LA')
    plt.plot(phase, RA, pen=mkPen(tuple(color_map["RA"]), width=3), symbol='o', symbolPen=tuple(color_map["RA"]),
             symbolBrush=0.2,
             name='RA')
    plt.plot(phase, RV, pen=mkPen(tuple(color_map["RV"]), width=3), symbol='o', symbolPen=tuple(color_map["RV"]),
             symbolBrush=0.2,
             name='RV')
    plt.plot(phase, LAA, pen=mkPen(tuple(color_map["LAA"]), width=3), symbol='o', symbolPen=tuple(color_map["LAA"]),
             symbolBrush=0.2,
             name='LAA')
    plt.plot(phase, SVC, pen=mkPen(tuple(color_map["SVC"]), width=3), symbol='o', symbolPen=tuple(color_map["SVC"]),
             symbolBrush=0.2,
             name='SVC')
    plt.plot(phase, IVC, pen=mkPen(tuple(color_map["IVC"]), width=3), symbol='o', symbolPen=tuple(color_map["IVC"]),
             symbolBrush=0.2,
             name='IVC')
    plt.plot(phase, PA, pen=mkPen(tuple(color_map["PA"]), width=3), symbol='o', symbolPen=tuple(color_map["PA"]),
             symbolBrush=0.2,
             name='PA')
    plt.plot(phase, PV, pen=mkPen(tuple(color_map["PV"]), width=3), symbol='o', symbolPen=tuple(color_map["PV"]),
             symbolBrush=0.2,
             name='PV')
    plt.plot(phase, LVM, pen=mkPen(tuple(color_map["LVM"]), width=3), symbol='o', symbolPen=tuple(color_map["LVM"]),
             symbolBrush=0.2,
             name='LVM')

    exporters.Exporter.Exporters.pop()
    CSVExporter.register()


def show():
    import sys
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec()

# ## Start Qt event loop.
# if __name__ == '__main__':
#     import sys
#
#     plot(None)
#     if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
#         pg.QtGui.QApplication.exec()
