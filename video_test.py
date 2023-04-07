import cv2

cap = cv2.VideoCapture(0)
# 設定擷取影像的尺寸大小
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(cv2.getBuildInformation())
#Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi', fourcc, 30.0, (1280,720),3)

while cap.isOpened():
    ret,frame = cap.read()
    #if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #水平上下反轉影像
    #frame = cv2.flip(frame,0)
    #write the flipped frame
    
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGRA2GRAY)
    cv2.imshow('frame',frame)
    out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
#Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows