import numpy as np
import objtracker
from objdetector import Detector
import cv2

VIDEO_PATH = './video/school1.mp4'
RESULT_PATH = 'result.mp4'

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Detections:
    def __init__(self):
        self.detections = []

    def add(self, xyxy, confidence, class_id, tracker_id):
        self.detections.append((xyxy, confidence, class_id, tracker_id))

def draw_trail(output_image_frame, trail_points, trail_color, trail_length=50):
    for i in range(len(trail_points)):
        if len(trail_points[i]) > 1:
            for j in range(1, len(trail_points[i])):
                cv2.line(output_image_frame, (int(trail_points[i][j-1][0]), int(trail_points[i][j-1][1])),
                         (int(trail_points[i][j][0]), int(trail_points[i][j][1])), trail_color[i], thickness=3)
        if len(trail_points[i]) > trail_length:
            trail_points[i].pop(0)  # Remove the oldest point from the trail

if __name__ == '__main__':
    # Initialize video capture to get video properties
    capture = cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file.")
        exit()

    # Get video properties (width and height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Close the video capture
    capture.release()

    detector = Detector()
    capture = cv2.VideoCapture(VIDEO_PATH)
    videoWriter = None
    fps = int(capture.get(5))
    print('fps:', fps)

    # Dictionary to store the trail points of each object
    object_trails = {}

    while True:
        _, im = capture.read()
        if im is None:
            break

        detections = Detections()
        output_image_frame, list_bboxs = objtracker.update(detector, im)

        for item_bbox in list_bboxs:
            x1, y1, x2, y2, _, track_id = item_bbox
            detections.add((x1, y1, x2, y2), None, None, track_id)

        # Add the current object's position to the trail
        for xyxy, _, _, track_id in detections.detections:
            x1, y1, x2, y2 = xyxy
            center = Point(x=(x1+x2)/2, y=(y1+y2)/2)
            if track_id in object_trails:
                object_trails[track_id].append((center.x, center.y))
            else:
                object_trails[track_id] = [(center.x, center.y)]

        # Draw the trail for each object
        trail_colors = [(255, 0, 255)] * len(object_trails)  # Red color for all trails
        draw_trail(output_image_frame, list(object_trails.values()), trail_colors)

        # Remove trails of objects that are not detected in the current frame
        for tracker_id in list(object_trails.keys()):
            if tracker_id not in [item[3] for item in detections.detections]:
                object_trails.pop(tracker_id)
        
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (output_image_frame.shape[1], output_image_frame.shape[0]))

        videoWriter.write(output_image_frame)
        cv2.imshow('Demo', output_image_frame)
        cv2.waitKey(1)

    capture.release()
    videoWriter.release()
    cv2.destroyAllWindows()



























