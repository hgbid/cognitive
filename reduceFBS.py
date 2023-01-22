import cv2

video = cv2.VideoCapture("pro.avi")

if (video.isOpened() == False):
    print("Error reading video file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('proRed.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)
i = 0
while (True):
    ret, frame = video.read()
    if ret == True:
        if i % 5 == 0:
            result.write(frame)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break

video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
