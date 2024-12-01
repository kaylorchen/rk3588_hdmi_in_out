import cv2

# 打开 /dev/video0 设备
cap = cv2.VideoCapture('/dev/video0')

# 检查设备是否成功打开
if not cap.isOpened():
    print("无法打开视频设备")
    exit()

while True:
    ret, frame = cap.read()  # 从设备读取一帧
    if not ret:
        print("无法读取摄像头数据")
        break

    cv2.imshow('Video Feed', frame)  # 显示视频帧

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()