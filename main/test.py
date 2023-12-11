import cv2
#摄像头IP地址
ip='10.112.89.10'
#摄像头登录用户名及密码q
user='admin'
password='a1234567'
cap = cv2.VideoCapture("rtsp://"+ user +":"+ password +"@" + ip + ":554/h264/ch1/main/av_stream")
# cap = cv2.VideoCapture("rtsp://"+ user +":"+ password +"@" + ip + ":554/h264/ch1/main/av_stream")
ret, frame = cap.read()
cv2.namedWindow(ip,0)
#窗体大小在这设置/ip为窗体显示名称  如需修改  参考 "名称"
cv2.resizeWindow(ip,500,300)
while ret:
    ret, frame = cap.read()
    cv2.imshow(ip,frame)
    #按下q键关闭窗体
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()