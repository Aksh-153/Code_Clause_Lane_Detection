import cv2
import numpy as np

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def resize_frame(frame):
    return cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = frame.shape[:2]
    roi_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.int32([roi_vertices]), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)

    line_image = np.zeros_like(frame)
    draw_lines(line_image, lines)

    output = cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)

    return output


def draw_lines(image, lines, color=(0, 255, 0), thickness=7):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


video_path = 'test2.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

frame = resize_frame(frame)

while ret:
    output_frame = detect_lanes(frame)

    cv2.imshow('Lane Detection', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

    frame = resize_frame(frame)

cap.release()
cv2.destroyAllWindows()
