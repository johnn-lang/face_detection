import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from qdrant import Qdrant
def facenet_preprocess(face_bgr):
    face = cv2.resize(face_bgr, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5          # [-1, 1]
    face = np.transpose(face, (2, 0, 1))  # CHW
    face = np.expand_dims(face, axis=0)   # (1,3,160,160)
    return face
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
class YoloDetector:
    def __init__(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Load YOLO
        self.model = YOLO(os.path.join(BASE_DIR, "best.pt"))

        # Load FaceNet ONNX
        self.session = ort.InferenceSession(
            os.path.join(BASE_DIR, "facenet.onnx"),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Load vector DB
        self.qdrant1 = Qdrant()
        self.db = self.qdrant1.extract_all()

        self.id = None
        self.no_face = False

    def detect(self, frame, register=False):
        self.id = None
        self.no_face = False

        results = self.model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            self.no_face = True
            return False

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.frame = frame

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_input = facenet_preprocess(face)

            emb = self.session.run(None, {self.input_name: face_input})[0]
 
            if register:
                self.embedding = emb
                return True

            max_sim = 0
            best_id = None

            for point in self.db:
                if point.payload["status"] == "Disabled":
                    continue

                sim = cosine_similarity(emb, np.array(point.vector))
                if sim > max_sim:
                    max_sim = sim
                    best_id = point.payload["group_id"]

            if max_sim >= 0.6:
                self.id = best_id
           
                return True

        return False
