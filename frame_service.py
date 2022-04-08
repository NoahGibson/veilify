import cv2

from face_landmark_service import detect_facial_landmarks, get_size


def overlay(image, filter_image, face_landmarks, INDEXES):
    '''
    This function will overlay a filter image over a face part of a person in the image/frame.
    Args:
        image:          The image of a person on which the filter image will be overlayed.
        filter_img:     The filter image that is needed to be overlayed on the image of the person.
        face_landmarks: The facial landmarks of the person in the image.
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

    except Exception:
        pass

    return annotated_image


def generate_frame_with_face_mesh(frame, face_mesh):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame, _ = detect_facial_landmarks(frame, face_mesh)
    return frame


def generate_frame_with_overlays(frame, face_mesh, overlays):
    _, face_mesh_results = detect_facial_landmarks(frame, face_mesh)

    annotated_frame = frame.copy()

    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            for overlay_image, indices in overlays:
                annotated_frame = overlay(
                    annotated_frame,
                    overlay_image,
                    face_landmarks,
                    indices
                )

    return annotated_frame
