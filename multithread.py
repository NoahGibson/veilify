from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

import traceback
import sys

import cv2
import numpy as np
import mediapipe as mp

from frame_service import generate_frame_with_face_mesh, generate_frame_with_overlays


class ThreadSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class WebcamThreadSignals(ThreadSignals):
    change_pixmap_signal = pyqtSignal(np.ndarray)


class WebcamThread(QThread):

    def __init__(self, fn, *args, **kwargs):
        super(WebcamThread, self).__init__()
        self.run_flag = False
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WebcamThreadSignals()

        self.kwargs["change_pixmap_callback"] = self.signals.change_pixmap_signal


    def run(self):
        self.run_flag = True

        try:
            result = self.fn(
                *self.args,
                **self.kwargs,
            )
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


    def stop(self):
        self.run_flag = False
        self.wait()


class WebcamWindow(QWidget):

    def __init__(self, *args, **kwargs):
        super(WebcamWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Veilify Webcam")
        self.display_width = 640
        self.display_height = 480
        self.setFixedSize(self.display_width, self.display_height)

        self.video_container = QLabel(self)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.video_container)

        self.setLayout(vbox)


    def update_frame(self, frame):
        qt_frame = self.convert_cv_to_qt(frame)
        self.video_container.setPixmap(qt_frame)


    def convert_cv_to_qt(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def update_display_size(self, width, height):
        self.display_width = width
        self.display_height = height
        self.setFixedSize(self.display_width, self.display_height)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Veilify")

        self.video_flip = True
        self.overlay_model_images = {}
        self.overlay_model_index = 0
        self.overlay_models = [
            "none",
            "face_mesh",
            "real_sample",
            "creep_0",
            "mustache"
        ]

        self.available_cameras = self.get_available_cameras()
        self.current_camera_index = 0

        self.overlay_label = QLabel("Overlay:")
        self.overlay_model_select = QComboBox()
        self.overlay_model_select.addItems(self.overlay_models)
        self.overlay_model_select.currentIndexChanged.connect(self.set_current_overlay_model)
        self.overlay_option = QHBoxLayout()
        self.overlay_option.addWidget(self.overlay_label)
        self.overlay_option.addWidget(self.overlay_model_select)

        self.camera_label = QLabel("Camera:")
        self.camera_select = QComboBox()
        self.camera_select.addItems(self.available_cameras)
        self.camera_select.currentIndexChanged.connect(self.set_current_camera)
        self.camera_option = QHBoxLayout()
        self.camera_option.addWidget(self.camera_label)
        self.camera_option.addWidget(self.camera_select)

        self.flip_video_toggle = QRadioButton("Flip Video")
        self.flip_video_toggle.setChecked(self.video_flip)
        self.flip_video_toggle.toggled.connect(self.set_video_flip)

        formbox = QVBoxLayout()
        formbox.addLayout(self.overlay_option)
        formbox.addLayout(self.camera_option)
        formbox.addWidget(self.flip_video_toggle)

        w = QWidget()
        w.setLayout(formbox)

        self.setCentralWidget(w)

        self.webcam_window = WebcamWindow()
        self.webcam_window_width = 640
        self.webcam_window_height = 480
        self.webcam_window.update_display_size(self.webcam_window_width, self.webcam_window_height)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

        self.webcam_thread = WebcamThread(self.webcam_loop)
        self.webcam_thread.signals.change_pixmap_signal.connect(self.update_frame)
        self.webcam_thread.start()

        self.webcam_window.show()


    def update_frame(self, frame):
        self.webcam_window.update_frame(frame)


    def webcam_loop(self, change_pixmap_callback):
        cap = cv2.VideoCapture(self.current_camera_index)
        cap.set(3, self.webcam_window_width)
        cap.set(4, self.webcam_window_height)

        while self.webcam_thread.run_flag:
            success, frame = cap.read()

            if not success:
                break

            if self.video_flip:
                frame = cv2.flip(frame, 1)

            overlay_model_name = self.get_current_overlay_model()

            if overlay_model_name == "none":
                pass
            elif overlay_model_name == "face_mesh":
                frame = generate_frame_with_face_mesh(
                    frame=frame,
                    face_mesh=self.face_mesh
                )
            else:
                frame = generate_frame_with_overlays(
                    frame=frame,
                    face_mesh=self.face_mesh,
                    overlays=self.get_overlays_for_model(overlay_model_name)
                )

            change_pixmap_callback.emit(frame)

        cap.release()


    def closeEvent(self, event):
        self.webcam_thread.stop()
        self.webcam_window.close()
        event.accept()


    def get_current_overlay_model(self):
        return self.overlay_models[self.overlay_model_index]


    def set_current_overlay_model(self, i):
        self.overlay_model_index = i


    def set_current_camera(self, i):
        self.current_camera_index = i
        self.webcam_thread.stop()
        self.webcam_thread.start()


    def set_video_flip(self):
        self.video_flip = self.flip_video_toggle.isChecked()


    def get_overlays_for_model(self, model_name):
        if model_name in self.overlay_model_images:
            return self.overlay_model_images[model_name]
        else:
            overlay_path = f"overlays/{model_name}"
            images = [
                (cv2.imread(f"{overlay_path}/mouth.png"), self.mp_face_mesh.FACEMESH_LIPS),
                (cv2.imread(f"{overlay_path}/left_eye.png"), self.mp_face_mesh.FACEMESH_LEFT_EYE),
                (cv2.imread(f"{overlay_path}/right_eye.png"), self.mp_face_mesh.FACEMESH_RIGHT_EYE)
            ]

            self.overlay_model_images[model_name] = images
            return images


    def get_available_cameras(self):
        return map(lambda camera: camera.description(), QCameraInfo.availableCameras())


if __name__=="__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
