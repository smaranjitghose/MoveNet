import os

import cv2
import numpy as np
import tensorflow as tf
from moviepy.editor import *


def main():

    # Dictionary that maps from joint names to keypoint indices.
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    # Maps bones to a matplotlib color name.
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): (255, 0, 255),
        (0, 2): (0, 255, 255),
        (1, 3): (255, 0, 255),
        (2, 4): (0, 255, 255),
        (0, 5): (255, 0, 255),
        (0, 6): (0, 255, 255),
        (5, 7): (255, 0, 255),
        (7, 9): (255, 0, 255),
        (6, 8): (0, 255, 255),
        (8, 10): (0, 255, 255),
        (5, 6): (255, 255, 0),
        (5, 11): (255, 0, 255),
        (6, 12): (0, 255, 255),
        (11, 12): (255, 255, 0),
        (11, 13): (255, 0, 255),
        (13, 15): (255, 0, 255),
        (12, 14): (0, 255, 255),
        (14, 16): (0, 255, 255),
    }

    # Load model
    model = tf.lite.Interpreter(
        model_path="./models/movenet_singlepose_lightning.tflite"
    )
    model.allocate_tensors()

    # Define path for output video file.
    output_file_path = "assets/output/result.avi"

    # Define path for input video file.
    input_file_path = "assets/input/demo.mp4"

    # Get video input from primary camera
    cap = cv2.VideoCapture(input_file_path)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    # Get dimesions of frame
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file_path, fourcc, 20.0, (frame_width, frame_height))

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            img = frame.copy()
            # Resize the frame
            img = tf.image.resize_with_pad(
                np.expand_dims(img, axis=0), target_height=192, target_width=192
            )
            # Type convert to tf.float32
            img = tf.cast(img, dtype=tf.float32)
            # Set up input and output formats for image
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            # Pass the frame in the desired input format
            model.set_tensor(input_details[0]["index"], np.array(img))
            # Make the predictions
            model.invoke()
            # Get the keypoints with confindence scores in the desired output format
            keypoints_with_scores = model.get_tensor(output_details[0]["index"])
            # Draw the edges
            draw_connections(
                frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR, 0.4
            )
            # Draw the keypoints
            draw_keypoints(frame, keypoints_with_scores, confidence_threshold=0.4)
            # Display the resulting frame
            cv2.imshow("Pose Estimation", frame)
            # Write the frame to a video file
            out.write(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        # Break the loop
        else:
            break

    # Release video capture object
    cap.release()
    # Close all windows
    cv2.destroyAllWindows()
    # Release video writer
    out.release()
    print("Initial .avi file saved to ./assets/output/result.avi")
    # Convert the file to .mp4 and remove .avi file
    convert_to_mp4(output_file_path)


def draw_keypoints(frame, keypoints, confidence_threshold=0.5):
    """
    Method to draw the appropriate keypoints for each frame
    """
    # Get the dimensions of the frame
    y, x, _ = frame.shape
    # Normalizing the Keypoint Coordinates according to the image size
    new_keypoints = np.squeeze(np.multiply(keypoints, np.array([y, x, 1])))
    print(new_keypoints)
    for keypoint in new_keypoints:
        # Getting the coordinates and the confidence coordinates for each keypoint
        ky, kx, kp_conf = keypoint
        # if the confidence score is less than the threshold, just ignore the keypoint detection altogether.
        if kp_conf > confidence_threshold:
            # Draw a circle filled with green color at the keypoint location with a radius of 3
            cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold=0.5):
    """
    Method to draw the edges between the appropriate keypoints for each frame
    """
    # Get the dimensions of the frame
    y, x, _ = frame.shape
    # Normalizing the Keypoint Coordinates according to the image size
    new_keypoints = np.squeeze(np.multiply(keypoints, np.array([y, x, 1])))

    for vertices, edge_color in edges.items():
        # Grab the vertices for a particular edge
        v1, v2 = vertices
        # Get the coordinates and confidence score for first vertex
        y1, x1, c1 = new_keypoints[v1]
        # Get the coordinates and confidence score for second vertex
        y2, x2, c2 = new_keypoints[v2]
        # Check if the confidence score of both vertices is above the required threshold
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            # Draw line of width 2 from the first vertex to the second vertex
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), edge_color, 2)


def convert_to_mp4(out_file_path: str) -> None:
    """
    Method that takes the .avi file and converts it to mp4 file with the same name.
    """
    nw_out_file_path = out_file_path.replace(".avi", ".mp4")
    clip = VideoFileClip(out_file_path)
    clip.write_videofile(nw_out_file_path)
    print("Removing the .avi file....")
    os.remove(out_file_path)


if __name__ == "__main__":
    main()
