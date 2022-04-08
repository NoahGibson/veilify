import cv2
import itertools
import numpy as np
import mediapipe as mp


def detect_facial_landmarks(image, face_mesh):
    '''
    This function performs facial landmarks detection on an image.
    Args:
        image:      The input image of person(s) whose facial landmarks needs to be detected.
        face_mesh:  The face landmarks detection function required to perform the landmarks detection.
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

    mp_face_mesh = mp.solutions.face_mesh

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
