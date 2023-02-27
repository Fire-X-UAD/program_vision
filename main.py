import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import math
import serial

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=0.7)

    def load_model(self):

        model = YOLO(".\\omni1.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame, iou=0.5, conf=0.25)

        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id
                       in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=self.labels)

        return frame

    def extract_data(self, results):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        return boxes, scores, class_ids

    def __call__(self):

        def lerp(a: float, b: float, t: float) -> float:
            """Linear interpolate on the scale given by a to b, using t as the point on that scale.
            Examples
            --------
                50 == lerp(0, 100, 0.5)
                4.2 == lerp(1, 5, 0.8)
            """
            return (1 - t) * a + t * b

        # cap = cv2.VideoCapture(self.capture_index)

        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # cap.set(cv2.CAP_PROP_FOCUS, 70)

        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        font = cv2.FONT_HERSHEY_SIMPLEX

        last_box = None
        buffered_frames = 0

        buffer_y1 = None
        buffer_x1 = None
        buffer_x2 = None
        buffer_y2 = None

        buffer_radius = 100
        manual_tracking_speed = 0.2

        posisi_bola = (None, None)

        # ser = serial.Serial('COM10', 9600, timeout=0, parity=serial.PARITY_NONE, rtscts=1)

        while True:
            start_time = time()
            ret, frame = cap.read()

            # # decrease frame size with lanczos4 interpolation
            # frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LANCZOS4)

            assert ret
            results = self.predict(frame)
            combined_img = self.plot_bboxes(results, frame)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(combined_img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            boxes, scores, class_ids = self.extract_data(results)

            # Search the highest score from the detected objects
            max_score = 0
            max_score_index = 0
            for i in range(len(scores)):
                if scores[i] > max_score:
                    max_score = scores[i]
                    max_score_index = i

            # Draw a line from the center of the frame to the center of the higest score object
            if len(boxes) > 0:
                x1 = int(boxes[max_score_index][0])
                y1 = int(boxes[max_score_index][1])
                x2 = int(boxes[max_score_index][2])
                y2 = int(boxes[max_score_index][3])
                # cv2.line(combined_img, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                #          (int((x1 + x2) / 2), int((y1 + y2) / 2)), (0, 255, 0), 2, cv2.LINE_AA)

                posisi_bola = (int((x1 + x2) / 2), int((y1 + y2) / 2))


                last_box = boxes[max_score_index]
                buffered_frames = 0
                buffer_y1 = None
                buffer_x1 = None
                buffer_x2 = None
                buffer_y2 = None
            # If nothing detected, draw a line from the center of the frame to the center of the last detected object, for 10 frames
            elif last_box is not None and buffered_frames < 20:
                x1 = int(last_box[0])
                y1 = int(last_box[1])
                x2 = int(last_box[2])
                y2 = int(last_box[3])

                # Buffer coordinates
                if buffer_y1 == None:
                    buffer_y1 = int((y1 + y2) / 2) - int(buffer_radius / 2) if int((y1 + y2) / 2) - int(
                        buffer_radius / 2) > 0 else 0
                    buffer_x1 = int((x1 + x2) / 2) - int(buffer_radius / 2) if int((x1 + x2) / 2) - int(
                        buffer_radius / 2) > 0 else 0
                    buffer_x2 = int((x1 + x2) / 2) + int(buffer_radius / 2) if int((x1 + x2) / 2) + int(
                        buffer_radius / 2) < frame.shape[1] else frame.shape[1]
                    buffer_y2 = int((y1 + y2) / 2) + int(buffer_radius / 2) if int((y1 + y2) / 2) + int(
                        buffer_radius / 2) < frame.shape[0] else frame.shape[0]

                # Grab 300px from the center of the last detected object
                cropped_img = combined_img[buffer_y1:buffer_y2, buffer_x1:buffer_x2]

                # Draw a rectangle around the cropped image
                cv2.rectangle(combined_img, (buffer_x1, buffer_y1), (buffer_x2, buffer_y2), (0, 255, 255), 2)

                # Draw text says "Tracking"
                cv2.putText(combined_img, "Deteksi warna manual...", (buffer_x1, buffer_y1 - 10), font, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)

                # Detect orange object from the cropped image
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
                # lower_orange = (0, 100, 100)
                # upper_orange = (20, 255, 255)
                # mask = cv2.inRange(hsv, lower_orange, upper_orange)
                ORANGE_MIN = np.array([0, 92, 192], np.uint8)
                ORANGE_MAX = np.array([5, 255, 255], np.uint8)
                ORANGE_MIN2 = np.array([174, 92, 192], np.uint8)
                ORANGE_MAX2 = np.array([179, 255, 255], np.uint8)
                mask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
                mask2 = cv2.inRange(hsv, ORANGE_MIN2, ORANGE_MAX2)
                mask = cv2.bitwise_or(mask, mask2)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=3)
                # mask = cv2.GaussianBlur(mask, (3, 3), 0)
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Find the biggest contour (the orange object)
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)

                    if w > 2 and h > 2:
                        # Draw line from the center of the frame to the center of the orange object
                        # cv2.line(combined_img, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int((x1+x2)/2)+x+int(w/2), int((y1+y2)/2)+y+int(h/2)), (0, 255, 0), 2, cv2.LINE_AA)
                        # cv2.line(combined_img, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int((x1+x2)/2)+x+int(w/2), int((y1+y2)/2)+y+int(h/2)), (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(cropped_img, "Orange", (x, y), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                        # cv2.line(combined_img, (int(frame.shape[1]/2), int(frame.shape[0]/2)), (int((x1+x2)/2), int((y1+y2)/2)), (0, 255, 0), 2, cv2.LINE_AA)

                        # Draw line from the center of the frame to the center of the orange object
                        # cv2.line(combined_img, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                        #          (buffer_x1 + x + int(w / 2), buffer_y1 + y + int(h / 2)), (0, 255, 0), 2, cv2.LINE_AA)

                        posisi_bola = (buffer_x1 + x + int(w / 2), buffer_y1 + y + int(h / 2))

                        # Lerp buffer coordinates to the center of the orange object
                        buffer_x1_temp = int(
                            lerp(buffer_x1, buffer_x1 + x + int(w / 2) - int(buffer_radius / 2), manual_tracking_speed))
                        buffer_y1_temp = int(
                            lerp(buffer_y1, buffer_y1 + y + int(h / 2) - int(buffer_radius / 2), manual_tracking_speed))
                        buffer_x1 = buffer_x1_temp if buffer_x1_temp > 0 else 0
                        buffer_y1 = buffer_y1_temp if buffer_y1_temp > 0 else 0
                        buffer_x2 = buffer_x1 + buffer_radius if buffer_x1 + buffer_radius < frame.shape[1] else \
                        frame.shape[1]
                        buffer_y2 = buffer_y1 + buffer_radius if buffer_y1 + buffer_radius < frame.shape[0] else \
                        frame.shape[0]

                else: # Bola tidak terdeteksi
                    buffered_frames += 1
                    if buffered_frames > 5:
                        last_box = None
                        buffer_x1 = None
                        buffer_y1 = None
                        buffer_x2 = None
                        buffer_y2 = None
                        posisi_bola = (None, None)

            if posisi_bola[0] is not None and posisi_bola[1] is not None:
                # titik_tengah = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
                titik_tengah = (int(frame.shape[1] / 2), int(frame.shape[0]))
                cv2.line(combined_img, titik_tengah, (posisi_bola[0], posisi_bola[1]), (0, 255, 0), 2, cv2.LINE_AA)
                # count angle from posisi_bola to titik_tengah 
                angle = int(math.atan2(titik_tengah[1] - posisi_bola[1], titik_tengah[0] - posisi_bola[0]) * 180 / math.pi)

                # offset angle by -90 degrees without negative values
                angle = angle - 90 if angle > 90 else angle + 270

                # show angle to screen
                cv2.putText(combined_img, str(angle), (10, 300), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                angle = -1

            # ser.write(str(angle).encode()+b"\n")

            cv2.imshow('YOLOv8 Detection', combined_img)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=1)
detector()
