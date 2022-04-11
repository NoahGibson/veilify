import cv2
import mediapipe as mp

from frame_service import generate_frame_with_face_mesh, generate_frame_with_overlays


def get_overlays_for_model(model_name, overlay_model_images):
    if model_name in overlay_model_images:
        return overlay_model_images[model_name]
    else:
        overlay_path = f"overlays/{model_name}"
        images = [
            (cv2.imread(f"{overlay_path}/mouth.png"), mp_face_mesh.FACEMESH_LIPS),
            (cv2.imread(f"{overlay_path}/left_eye.png"), mp_face_mesh.FACEMESH_LEFT_EYE),
            (cv2.imread(f"{overlay_path}/right_eye.png"), mp_face_mesh.FACEMESH_RIGHT_EYE)
        ]

        overlay_model_images[model_name] = images
        return images


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 960)

window_name = "Veilify"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

flip_camera = False

overlay_model_images = {}
overlay_model_index = 1
overlay_models = [
    "face_mesh",
    "real_sample",
    "creep_0",
    "mustache"
]

while capture.isOpened():
    success, frame = capture.read()

    if not success:
        continue

    if flip_camera:
        frame = cv2.flip(frame, 1)

    overlay_model_name = overlay_models[overlay_model_index]
    if overlay_model_name == "face_mesh":
        frame = generate_frame_with_face_mesh(
            frame=frame,
            face_mesh=face_mesh
        )
    else:
        frame = generate_frame_with_overlays(
            frame=frame,
            face_mesh=face_mesh,
            overlays=get_overlays_for_model(overlay_model_name, overlay_model_images)
        )

    cv2.imshow(window_name, frame)

    k = cv2.waitKey(1) & 0xFF

    if (k == ord("q")):
        break

    if (k == ord("f")):
        flip_camera = not flip_camera

    if (k == ord(" ")):
        overlay_model_index += 1
        overlay_model_index = overlay_model_index if overlay_model_index < len(overlay_models) else 0

capture.release()
cv2.destroyAllWindows()
