import glob
import os

import cv2
import mediapipe as mp
import np
from tkinter import filedialog as fd

import xlsxwriter


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


folder = fd.askdirectory()

for vid_file in os.listdir(folder):
    f = open(vid_file[:-4] +".txt", "w")
    cell_n = '1'
    workbook = xlsxwriter.Workbook(vid_file[:-4]+'.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', "T (sec)")
    worksheet.write('B1', "leftHip")
    worksheet.write('C1', "shoulder")
    worksheet.write('D1', "elbow")
    worksheet.write('E1', "LH-S-E angle")
    worksheet.write('F1', "S-E-W angle")
    worksheet.write('G1', "E-W-F angle")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(folder+'\\'+vid_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            # Make detection
            results = pose.process(image)

            try:
                landmarks = results.pose_landmarks.landmark

                leftHip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                fingers = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]

                cell_n = str(int(cell_n) + 1)
                worksheet.write('A'+cell_n, str(duration))
                worksheet.write('B'+cell_n, str(leftHip))
                worksheet.write('C'+cell_n, str(shoulder))
                worksheet.write('D'+cell_n, str(elbow))

                f.write("T (sec) = " + str(duration) + '\n')
                f.write("leftHip: " + str(leftHip) + '\n')
                f.write("shoulder: " + str(shoulder) + '\n')
                f.write("elbow: " + str(elbow) + '\n')

                angle = calculate_angle(leftHip, shoulder, elbow)
                angle1 = calculate_angle(shoulder, elbow, wrist)
                angle2 = calculate_angle(elbow, wrist, fingers)
                f.write("LH-S-E angle: " + str(angle))
                f.write("S-E-W angle: " + str(angle1))
                f.write("E-W-F angle: " + str(angle2))
                worksheet.write('E'+cell_n, str(angle))
                worksheet.write('F'+cell_n, str(angle1))
                worksheet.write('G'+cell_n, str(angle2))

            except:
                pass

    f.close()
    workbook.close()
    cap.release()
    cv2.destroyAllWindows()
print("DONE")

import time


