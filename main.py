import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt


def detect_facial_landmarks(image, face_mesh, display=True):
    '''
    This function performs facial landmarks detection on an image.
    Args:
        image:      The input image of person(s) whose facial landmarks needs to be detected.
        face_mesh:  The face landmarks detection function required to perform the landmarks detection.
        display:    A boolean value that, if set to true, has the function display the original input image
                    and the output image with the face landmarks drawn and returns nothing.
    Returns:
        output_image:   A copy of input image with face landmarks drawn.
        results:        The output of the facial landmarks detection on the input image.
    '''

    # Perform facial landmark detection
    results = face_mesh.process(image)

    # Create copy of input image to draw facial landmarks on
    output_image = image.copy()

    # Check if facial landmarks were found
    if results.multi_face_landmarks:

        # Iterate over found faces
        for face_landmarks in results.multi_face_landmarks:

            # Draw the face mesh tesselation on the output image
            mp.solutions.drawing_utils.draw_landmarks(
                image=output_image,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Check if the original input and output iamge are to be displayed
    if display:
        plt.figure(figsize=[15, 15])

        plt.subplot(121)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(output_image)
        plt.title("Output")
        plt.axis("off")
    else:
        return np.ascontiguousarray(output_image[:,:,::-1], dtype=np.uint8), results


def get_size(image, face_landmarks, INDEXES):
    '''
    This function calculates the height and width of a face part utilizing its landmarks.
    Args:
        image:          The image of person(s) whose face part size is to be calculated.
        face_landmarks: The detected face landmarks of the person whose face part size is to
                        be calculated.
        INDEXES:        The indexes of the face part landmarks, whose size is to be calculated.
    Returns:
        width:     The calculated width of the face part of the face whose landmarks were passed.
        height:    The calculated height of the face part of the face whose landmarks were passed.
        landmarks: An array of landmarks of the face part whose size is calculated.
    '''

    # Retrieve height and width of image
    image_height, image_width, _ = image.shape

    # Convert the indexes of the landmarks of the face parts into a list
    INDEXES_LIST = list(itertools.chain(*INDEXES))

    # Initialize a list to store the landmarks of the face part
    landmarks = []

    # Iterate over the indexes of the landmarks
    for INDEX in INDEXES_LIST:
        landmarks.append(
            [
                int(face_landmarks.landmark[INDEX].x * image_width),
                int(face_landmarks.landmark[INDEX].y * image_height)
            ]
        )

    # Convert the list of landmarks into a numpy array
    landmarks = np.array(landmarks)

    # Calculate the width and height of the face part
    _, _, width, height = cv2.boundingRect(landmarks)

    # Return the width and height and the landmarks
    return width, height, landmarks


def is_open(image, face_mesh_results, face_part, threshold=5):
    '''
    This function checks whether the an eye or mouth of the person(s) is open,
    utilizing its facial landmarks.
    Args:
        image:             The image of person(s) whose an eye or mouth is to be checked.
        face_mesh_results: The output of the facial landmarks detection on the image.
        face_part:         The name of the face part that is required to check.
        threshold:         The threshold value used to check the isOpen condition.
    Returns:
        status:       A dictionary containing isOpen statuses of the face part of all the
                      detected faces.
    '''

    # Create a dict to store the status of each face part
    status = {}

    # Check what type of face part it is
    if face_part == "MOUTH":

        # Get indexes of mouth
        INDEXES = mp_face_mesh.FACEMESH_LIPS

    elif face_part == "LEFT_EYE":

        # Get indexes of left eye
        INDEXES = mp_face_mesh.FACEMESH_LEFT_EYE

    elif face_part == "RIGHT_EYE":

        # Get indexes of right eye
        INDEXES = mp_face_mesh.FACEMESH_RIGHT_EYE

    else:
        return

    # Iterate over the found faces
    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        # Get height of the part
        _, height, _ = get_size(image, face_landmarks, INDEXES)

        # Get height of the entire face
        _, face_height, _ = get_size(image, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL)

        # Check if part is open
        if (height / face_height) * 100 > threshold:
            status[face_no] = "OPEN"
        else:
            status[face_no] = "CLOSE"

    return status


def overlay(image, filter_image, face_landmarks, face_part, INDEXES):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    Args:
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
        face_part:      The name of the face part on which the filter image will be overlayed.
        INDEXES:        The indexes of landmarks of the face part.
        display:        A boolean value that is if set to true the function displays
                        the annotated image and returns nothing.
    Returns:
        annotated_image: The image with the overlayed filter on the top of the specified face part.
    '''

    # Create copy of the image
    annotated_image = image.copy()

    # Errors can come when it resizes the filter image to a too small or a too large size .
    # So use a try block to avoid application crashing
    try:

        # Get the width and height of the filter image
        filter_image_height, filter_image_width, _ = filter_image.shape

        # Get the height of the face part on which to overlay the filter
        _, face_part_height, landmarks = get_size(image, face_landmarks, INDEXES)

        # Specify the height to which the filter image is required to be resized
        required_height = int(face_part_height * 3)

        # Resize the filter image
        resized_filter_image = cv2.resize(
            filter_image,
            (int(filter_image_width * (required_height / filter_image_height)), required_height)
        )

        # Get the new width and height of the filter
        filter_image_height, filter_image_width, _ = resized_filter_image.shape

        # Convert the image to grayscale and apply threshold to get mask image
        _, filter_image_mask = cv2.threshold(
            cv2.cvtColor(resized_filter_image, cv2.COLOR_BGR2GRAY),
            25,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Calculate the center of the part
        center = landmarks.mean(axis=0).astype("int")

        # Calculate location to place filter
        location = (int(center[0] - filter_image_width / 2), int(center[1] - filter_image_height / 2))

        # Retrieve region of interest from the image where filter will be placed
        ROI = image[
            location[1]: location[1] + filter_image_height,
            location[0]: location[0] + filter_image_width
        ]

        # Perform bitwase AND op to set pixel values where the filter will be placed to 0
        resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_image_mask)

        # Add the resultant image and the filter
        resultant_image = cv2.add(resultant_image, resized_filter_image)

        # Update the image's region of interest to the resultant
        annotated_image[
            location[1]: location[1] + filter_image_height,
            location[0]: location[0] + filter_image_width
        ] = resultant_image

    except Exception as e:
        pass

    return annotated_image


def face_landmarks_detection(cap):
    cv2.namedWindow("Face Landmarks Detection", cv2.WINDOW_NORMAL)

    time1=0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, _ = detect_facial_landmarks(frame, face_mesh_videos, display=False)

        time2 = time()

        if (time2 - time1) > 0:
            fps = 1.0 / (time2 - time1)
            cv2.putText(
                frame,
                "FPS: {}".format(int(fps)),
                (10, 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                3
            )

        time1 = time2

        cv2.imshow("Face Landmarks Detection", frame)

        k = cv2.waitKey(1) & 0xFF

        if (k == ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def snapchat_filter_webcam(cap):
    cv2.namedWindow("Face Filter", cv2.WINDOW_NORMAL)

    eye = cv2.imread("media/eye.png")
    left_eye = cv2.imread("media/left_eye.png")
    right_eye = cv2.imread("media/right_eye.png")
    mouth = cv2.imread("media/smile.png")

    flip_camera = False

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            continue

        if flip_camera:
            frame = cv2.flip(frame, 1)

        _, face_mesh_results = detect_facial_landmarks(frame, face_mesh_videos, display=False)

        if face_mesh_results.multi_face_landmarks:
            left_eye_status = is_open(
                frame,
                face_mesh_results,
                "LEFT_EYE",
                threshold=4.5
            )

            right_eye_status = is_open(
                frame,
                face_mesh_results,
                "RIGHT_EYE",
                threshold=4.5
            )

            for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                frame = overlay(
                    frame,
                    mouth,
                    face_landmarks,
                    "MOUTH",
                    mp.solutions.face_mesh.FACEMESH_LIPS
                )

                # if left_eye_status[face_num] == "OPEN":
                frame = overlay(
                    frame,
                    eye,
                    face_landmarks,
                    "LEFT_EYE",
                    mp.solutions.face_mesh.FACEMESH_LEFT_EYE
                )
                # if right_eye_status[face_num] == "OPEN":
                frame = overlay(
                    frame,
                    eye,
                    face_landmarks,
                    "RIGHT_EYE",
                    mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
                )



        cv2.imshow("Face Filter", frame)

        k = cv2.waitKey(1) & 0xFF

        if (k == ord("q")):
            break

        if (k == ord("f")):
            flip_camera = not flip_camera

    cap.release()
    cv2.destroyAllWindows()


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

sample_img = cv2.cvtColor(cv2.imread("media/sample.png"), cv2.COLOR_BGR2RGB)

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    min_detection_confidence=0.5
)

face_mesh_videos = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)

snapchat_filter_webcam(camera_video)