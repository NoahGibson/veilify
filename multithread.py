from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import traceback
import sys

import cv2
import numpy as np


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    change_pixmap_signal = pyqtSignal(np.ndarray)


class Worker(QRunnable):

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        self.kwargs["change_pixmap_callback"] = self.signals.change_pixmap_signal

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(
                *self.args,
                **self.kwargs
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result) # Return the result of the processing
        finally:
            self.signals.finished.emit() # Done


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Veilify")
        self.display_width = 1280
        self.display_height = 960

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        self.textLabel = QLabel("Webcam")

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        self.setLayout(vbox)

        self.threadpool = QThreadPool()

        worker = Worker(self.execute_this_fn)
        worker.signals.change_pixmap_signal.connect(self.update_image)

        self.webcam_running = True

        self.threadpool.start(worker)

    def update_image(self, frame):
        qt_frame = self.convert_cv_to_qt(frame)
        self.image_label.setPixmap(qt_frame)

    def execute_this_fn(self, change_pixmap_callback):
        cap = cv2.VideoCapture(0)
        while self.webcam_running:
            success, frame = cap.read()

            if success:
                change_pixmap_callback.emit(frame)

        cap.release()

    def convert_cv_to_qt(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.webcam_running = False
        event.accept()


if __name__=="__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
