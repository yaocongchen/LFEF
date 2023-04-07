import cv2

cap = cv2.VideoCapture('/home/yaocong/Experimental/speed_smoke_segmentation/smoke_video.mp4')
#Define the codec and create VideoWriter object

while cap.isOpened():
    ret,frame = cap.read()
    #if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows