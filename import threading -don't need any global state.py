import threading
import queue
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

reference_image = cv2.imread("reference_image.jpg")
if reference_image is None:
    raise FileNotFoundError("reference_image.jpg not found or cannot be opened.")

result_queue = queue.Queue()

def check_face(frame, reference_image, result_queue):
    try:
        match = DeepFace.verify(frame, reference_image.copy())['verified']
        result_queue.put(match)
    except Exception:
        result_queue.put(False)

face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 10 == 0:
            threading.Thread(target=check_face, args=(frame.copy(), reference_image, result_queue)).start()
            counter += 1

        # Get result from queue if available
        try:
            face_match = result_queue.get_nowait()
        except queue.Empty:
            pass

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break