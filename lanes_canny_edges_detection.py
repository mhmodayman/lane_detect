import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # speeds up processing
    blur = cv2.GaussianBlur(gray, (5,5), 0) # blurring or smoothing reduces noise
    canny = cv2.Canny(blur, 50, 150) # canny edge detection, note that above line is optional, because it is already implemented in canny
    return canny

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    msk1 = np.zeros_like(img)
    cv2.fillPoly(msk1, polygons, 255)
    msk2 = cv2.bitwise_and(img, msk1) # bitwise and operation to extract white lanes
    return msk2

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0] # height
    y2 = int(y1*(3/5)) # three fifth of the image from down to up
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # polynomial of degree 1
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])

"""
image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image) # not needed for now
canny_image = canny(image)
extracted_lanes = region_of_interest(canny_image)
straight_lines = cv2.HoughLinesP(extracted_lanes, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # we draw straight lines over extracted lanes

averaged_lines = average_slope_intercept(image, straight_lines)

combined = display_lines(image, averaged_lines) # this function returns image combined with straight lines that we draw above
combo = cv2.addWeighted(image, 0.8, combined, 1, 1)

#plt.imshow(canny)
#plt.show()

cv2.imshow('result', combo)
cv2.waitKey(0) # waits infinitely till we press any key
cv2.destroyWindow('result')
"""


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    extracted_lanes = region_of_interest(canny_image)
    straight_lines = cv2.HoughLinesP(extracted_lanes, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # we draw straight lines over extracted lanes

    averaged_lines = average_slope_intercept(frame, straight_lines)

    combined = display_lines(frame, averaged_lines) # this function returns image combined with straight lines that we draw above
    combo = cv2.addWeighted(frame, 0.8, combined, 1, 1)
    
    cv2.imshow('result', combo)
    if cv2.waitKey(1) & 0xFF == ord('q'): # waits 1 ms
        break
cap.release()
cv2.destroyAllWindows('result')