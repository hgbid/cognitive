import glob
import os
import numpy as np

import cv2
import mediapipe as mp
import np
from tkinter import filedialog as fd

import xlsxwriter

ROATE = False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


folder = fd.askdirectory()

for vid_file in os.listdir(folder):
    cell_n = '1'
    workbook = xlsxwriter.Workbook(vid_file[:-4] + '3D.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', "T (sec)")
    worksheet.write('B1', "LeftShoulderAngle")
    worksheet.write('C1', "LeftElbowAngle")
    worksheet.write('D1', "LeftWristAngle")
    worksheet.write('E1', "RightShoulderAngle")
    worksheet.write('F1',"RightElbowAngle")
    worksheet.write('G1',"RightWristAngle")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(folder + '\\' + vid_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with mp_pose.Pose(min_detection_confidence=0, min_tracking_confidence=0) as pose:
        print("File " + vid_file + " starting")
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            duration = frame_num / fps

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if ROATE:
                image = cv2.rotate(image, cv2.ROTATE_180)
            # Make detection
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark

                lh_val = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                leftHip = [lh_val.x, lh_val.y, lh_val.z]
                sh_val = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                leftShoulder = [sh_val.x, sh_val.y, sh_val.z]
                e_val = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                leftElbow = [e_val.x, e_val.y, e_val.z]
                w_val = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                leftWrist = [w_val.x, w_val.y, w_val.z]
                f_val = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                leftFingers = [f_val.x, f_val.y, f_val.z]

                rh_val = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                rightHip = [rh_val.x, rh_val.y, rh_val.z]
                RS_val = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                rightShoulder = [RS_val.x, RS_val.y, RS_val.z]
                RE_val = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                rightElbow = [RE_val.x, RE_val.y, RE_val.z]
                RW_val = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                rightWrist = [RW_val.x, RW_val.y, RW_val.z]
                RI_val = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                rightFingers = [RI_val.x, RI_val.y, RI_val.z]

                # Calculate angle
                leftShoulderAngle = calculate_angle(leftHip, leftShoulder, leftElbow)
                leftElbowAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
                leftWristAngle = calculate_angle(leftElbow, leftWrist, leftFingers)

                rightShoulderAngle = calculate_angle(rightHip, rightShoulder, rightElbow)
                rightElbowAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)
                rightWristAngle = calculate_angle(rightElbow, rightWrist, rightFingers)

                cell_n = str(int(cell_n) + 1)
                worksheet.write('A' + cell_n, str(duration))
                worksheet.write('B' + cell_n, str(leftShoulderAngle))
                worksheet.write('C'+cell_n, str(leftElbowAngle))
                worksheet.write('D'+cell_n, str(leftWristAngle))
                worksheet.write('E' + cell_n, str(rightShoulderAngle))
                worksheet.write('F'+cell_n, str(rightElbowAngle))
                worksheet.write('G'+cell_n, str(rightWristAngle))

            except:
                pass

    workbook.close()
    cap.release()
    cv2.destroyAllWindows()
print("DONE")
