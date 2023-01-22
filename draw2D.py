import cv2
import mediapipe as mp
import np
from functools import reduce

BLACK = (255, 0, 0)
THICKNESS = 2
ROATE = True


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 2)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
counter = 0
stage = None

cap = cv2.VideoCapture("video/proRed.avi")
size = (int(cap.get(3)), int(cap.get(4)))
result = cv2.VideoWriter('tabletDraw.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                         cap.get(cv2.CAP_PROP_FPS), size)

counter = 0
stage = None
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if ROATE:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = pose.process(image)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            lh_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            leftHip = [lh_val.x, lh_val.y]
            sh_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            leftShoulder = [sh_val.x, sh_val.y]
            e_val = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            leftElbow = [e_val.x, e_val.y]
            w_val = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            leftWrist = [w_val.x, w_val.y]
            f_val = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
            leftFingers = [f_val.x, f_val.y]

            rh_val = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            rightHip = [rh_val.x, rh_val.y]
            RS_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            rightShoulder = [RS_val.x, RS_val.y]
            RE_val = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            rightElbow = [RE_val.x, RE_val.y]
            RW_val = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            rightWrist = [RW_val.x, RW_val.y]
            RW_val = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
            rightFingers = [RW_val.x, RW_val.y]

            # Calculate angle
            leftShoulderAngle = calculate_angle(leftHip, leftShoulder, leftElbow)
            leftElbowAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
            leftWristAngle = calculate_angle(leftElbow, leftWrist, leftFingers)

            rightShoulderAngle = calculate_angle(rightHip, rightShoulder, rightElbow)
            rightElbowAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)
            rightWristAngle = calculate_angle(rightElbow, rightWrist, rightFingers)

            # Visualize angle
            cv2.putText(image, str(leftShoulderAngle),
                        tuple(np.multiply(leftShoulder, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, str(leftElbowAngle),
                        tuple(np.multiply(leftElbow, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, str(leftWristAngle),
                        tuple(np.multiply(leftWrist, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)

            cv2.putText(image, str(rightShoulderAngle),
                        tuple(np.multiply(rightShoulder, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, str(rightElbowAngle),
                        tuple(np.multiply(rightElbow, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)
            cv2.putText(image, str(rightWristAngle),
                        tuple(np.multiply(rightWrist, size).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, THICKNESS, cv2.LINE_AA)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('MediaPipe Pose', image)
        result.write(image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
result.release()

cv2.destroyAllWindows()
print("DONE")
