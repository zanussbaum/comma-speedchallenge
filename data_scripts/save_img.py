import cv2 as cv

cap = cv.VideoCapture("../video/train.mp4")

frame_count = 0
_ret = True
while _ret:
    _ret, frame = cap.read()
    if not _ret:
        continue
    frame_count += 1
    if frame_count % 1000 == 0:
        print(f"Finished {frame_count} frames")

    cv.imwrite(f"../frames/original/frame_{frame_count}.jpg", frame)
    

assert frame_count == 20400, "Missing frames"
