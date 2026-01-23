from flask import Flask, Response, render_template_string
import socket, struct, threading, time
import cv2
import numpy as np

app = Flask(__name__)

# ================== GLOBAL ==================
latest_frame = None
last_frame_time = 0
FRAME_TIMEOUT = 2  # seconds
frame_lock = threading.Lock()

# ================== FAKE VIDEO ==================
def fake_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for Raspberry Pi camera...",
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    return frame

# ================== SOCKET SERVER ==================
def camera_socket_server():
    global latest_frame, last_frame_time

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 9999))
    s.listen(1)

    print("📡 TCP camera server listening on port 9999")

    while True:
        conn, addr = s.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("🔗 Raspberry Pi connected:", addr)

        try:
            while True:
                raw_len = conn.recv(4)
                if not raw_len:
                    break

                frame_len = struct.unpack(">I", raw_len)[0]
                data = b""
                while len(data) < frame_len:
                    packet = conn.recv(frame_len - len(data))
                    if not packet:
                        break
                    data += packet

                frame = cv2.imdecode(
                    np.frombuffer(data, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if frame is not None:
                    with frame_lock:
                        latest_frame = frame   # overwrite frame cũ
                        last_frame_time = time.time()

        except Exception as e:
            print("❌ Socket error:", e)

        finally:
            conn.close()
            with frame_lock:
                latest_frame = None
            print("🔌 Raspberry Pi disconnected")

# ================== STREAM ==================
def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is not None and time.time() - last_frame_time < FRAME_TIMEOUT:
                
                frame = latest_frame.copy()
            else:
                frame = fake_frame()

        ret, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

        time.sleep(0.03)  # ~30 FPS

# ================== FLASK ==================
@app.route("/")
def index():
    return render_template_string("""
    <html>
        <body>
            <h2>📷 Raspberry Pi Camera Stream</h2>
            <img src="/video_feed">
        </body>
    </html>
    """)

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================== MAIN ==================
if __name__ == "__main__":
    threading.Thread(
        target=camera_socket_server,
        daemon=True
    ).start()

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        use_reloader=False
    )
