from picamera2 import Picamera2
import cv2
import socket
import struct
import time
import os


SERVER_IP = "10.3.64.253"  
SERVER_PORT = 9999

SAVE_FACE = True
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()


face_cascade = cv2.CascadeClassifier(
    "/home/thienvuong/haarcascade_frontalface_default.xml"
)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_IP, SERVER_PORT))
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

print("✅ Connected to server")


i = 0

try:
    while True:
        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )



        ret, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        if not ret:
            continue

        data = buffer.tobytes()

        # Send frame length + frame
        sock.sendall(struct.pack(">I", len(data)))
        sock.sendall(data)

        time.sleep(0.03)  # ~30 FPS

except KeyboardInterrupt:
    print("⛔ Stopped by user")

finally:
    sock.close()
    picam2.stop()
