import sys
import os
import time
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


"""
Google MediaPipe has an easy to use and fast hand tracking model - here we create a wrapper class to use it as an input
to our tasks.

The MediaPipe model outputs the joint positions of 21 hand joints. The code here then estimates a single "flexion" value
for each finger, which is a number between 0 (fully extended) and 1 (fully flexed). Using MP joint positions, we first 
calculate the angle of each finger joint, calculate the summed angle of all the joints in the finger, and then normalize
between the min and max sum. For example, with index finger, at 0% flexion all the joints (starting at the wrist) 
are straight, with a summed angle of 0 deg. At 100% flexion, all the joints are bent, with a summed angle 
around 250-270 deg. At 0 flexion, there's some error in the MP model such that the summed angle is 10-20 deg per joint
(and this error changes a little with hand rotation), which is why the min angles are not 0

** You may need to change the calibration range in the `finger_joint_min_angle` and `finger_joint_max_angle` dicts **

Notes:
- Reference for the tracked joints and their indices:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models
- Reference for the model result format:
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#handle_and_display_results
- In the future, may want to switch to the MediaPipe async function to improve performance
(see example https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/raspberry_pi/detect.py)

"""

# Create an index of the joints of each finger, and the min and max summed angle for each finger
finger_joint_indices = {
    'thumb': [0, 1, 2, 3, 4],
    'index': [0, 5, 6, 7, 8],
    'middle': [0, 9, 10, 11, 12],
    'ring': [0, 13, 14, 15, 16],
    'small': [0, 17, 18, 19, 20]
}

finger_joint_min_angle = {
    'thumb': 30,
    'index': 65,
    'middle': 50,
    'ring': 50,
    'small': 40
}

finger_joint_max_angle = {
    'thumb': 100,
    'index': 240,
    'middle': 250,
    'ring': 250,
    'small': 250
}


def angle_between_points(a, b, c):
    # helper function to calculate the angle at point b formed by points a, b, and c. Returns value in degrees
    vec_ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    vec_bc = (b.x - c.x, b.y - c.y, b.z - c.z)

    dot_prod = sum(u*v for u, v in zip(vec_ab, vec_bc))
    mag_ab = math.sqrt(sum(u**2 for u in vec_ab))
    mag_bc = math.sqrt(sum(u**2 for u in vec_bc))

    angle_radians = math.acos(dot_prod / (mag_ab * mag_bc))
    return min(math.degrees(angle_radians), 360 - math.degrees(angle_radians))


class HandTracker:
    def __init__(self, camera_id, show_tracking=False):
        self.do_show_tracking = show_tracking

        # set up the webcam
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        # initialize the hand tracking model
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(this_file_dir, 'models', 'mediapipe_hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.hand_tracker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
        )

    def get_hand_position(self):
        # get frame from camera
        success, image = self.camera.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        # flip image and convert from BGR to RGB
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # run the mediapipe hand tracker
        hand_result = self.hand_tracker.detect_for_video(mp_image, time.time_ns() // 1_000_000)

        # calc finger flexion from the tracked joint locations (if hand is detected)
        if hand_result and hand_result.hand_world_landmarks:
            finger_flex = self.calc_finger_flex(hand_result)
        else:
            finger_flex = [0, 0, 0, 0, 0]

        # optionally plot the tracking
        if self.do_show_tracking:
            self.draw_hand_tracking(image, hand_result, finger_flex)
            cv2.imshow('Hand Tracking', image)
            cv2.waitKey(1)

        return finger_flex

    @staticmethod
    def calc_finger_flex(hand_result):
        # go from finger joint locations to approximate % finger flexion for each finger

        world_landmarks = hand_result.hand_world_landmarks[0]    # idx 0 since we only have 1 hand

        flexions = []
        for fing in ['thumb', 'index', 'middle', 'ring', 'small']:
            joint_indices = finger_joint_indices[fing]
            joint_angles = []

            # find the angle at each joint. For example, use joints 0, 1, and 2 to find the angle at joint 1
            for i in range(len(joint_indices) - 2):
                joint_a = world_landmarks[joint_indices[i]]
                joint_b = world_landmarks[joint_indices[i + 1]]
                joint_c = world_landmarks[joint_indices[i + 2]]
                joint_angles.append(angle_between_points(joint_a, joint_b, joint_c))

            # sum the angles at each joint and normalize between 0 and 1
            flexion = sum(joint_angles)
            flexion = ((flexion - finger_joint_min_angle[fing]) /
                       (finger_joint_max_angle[fing] - finger_joint_min_angle[fing]))
            flexion = max(0, min(1, flexion))  # clamp between 0 and 1
            flexions.append(flexion)

        return flexions

    @staticmethod
    def draw_hand_tracking(image, hand_tracking_result, finger_flex):
        for hand_idx in range(len(hand_tracking_result.hand_landmarks)):
            hand_landmarks = hand_tracking_result.hand_landmarks[hand_idx]
            # handedness = hand_tracking_result.handedness[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # add a text label on each joint specifying the joint idx
            for joint_idx, joint in enumerate(hand_landmarks):
                cv2.putText(image,
                            str(joint_idx),
                            (int(joint.x * image.shape[1]), int(joint.y * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)

            # add a text label for each finger with the flexion value
            for finger_idx, flex in enumerate(finger_flex):
                cv2.putText(image,
                            f"{100*flex:.0f}%",
                            (int(0.3 * image.shape[1] + finger_idx * 0.06 * image.shape[1]), int(0.05 * image.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA)
