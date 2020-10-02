import cv2 as cv
import numpy as np
import math

from Project.project_1.utils.hu_moments_generation import hu_moments_of_file, generate_hu_moments_file
from Project.project_1.utils.label_converters import int_to_label
from Project.project_1.utils.training_model import train_model


def detection():
    generate_hu_moments_file()
    model = train_model()

    track = False
    cap = cv.VideoCapture(1)
    window_name = 'Binary'
    trackbar_name_1 = "Trackbar"
    cv.namedWindow(window_name)
    cv.createTrackbar(trackbar_name_1, window_name, 153, 255, on_trackbar_name)

    trackbar_name_2 = 'Min area'
    cv.createTrackbar(trackbar_name_2, window_name, 0, 5, on_trackbar_name)
    trackbar_name_3 = 'Max area'
    cv.createTrackbar(trackbar_name_3, window_name, 2, 20, on_trackbar_name)
    trackbar_name_4 = 'Error'
    cv.createTrackbar(trackbar_name_4, window_name, 1, 100, on_trackbar_name)
    saved_contour = None

    while True:
        _, frame_1 = cap.read()
        _, frame_2 = cap.read()

        thresh_value = cv.getTrackbarPos(trackbar_name_1, window_name)
        min_area = cv.getTrackbarPos(trackbar_name_2, window_name)
        max_area = cv.getTrackbarPos(trackbar_name_3, window_name)
        area_range = [min_area, max_area]
        error = cv.getTrackbarPos(trackbar_name_4, window_name)

        grayscale = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(grayscale, thresh_value, 255, cv.THRESH_BINARY)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        # figure we use to erode/dilate
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

        contours_noise, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contours_denoise, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        filtered_contours = filter_contours(contours_denoise, area_range)
        grayscale_2_bgr = cv.cvtColor(grayscale, cv.COLOR_GRAY2BGR)
        cv.drawContours(grayscale_2_bgr, filtered_contours, -1, (66, 66, 245), 3)
        # -1 draws ALL contours,

        # MACHINE LEARNING
        if filtered_contours:
            selected = max(filtered_contours, key=cv.contourArea)
            moments = cv.moments(selected)
            hu_moments = cv.HuMoments(moments)

            for i in range(0, 7):
                hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

            test_response = model.predict(np.array([hu_moments], dtype=np.float32))[1]
            image_with_text = cv.putText(grayscale_2_bgr, int_to_label(test_response), (50, 50), cv.FONT_HERSHEY_DUPLEX, 1,
                                         (245, 49, 118), 2, cv.LINE_AA)
        ############

        if track:
            for contour in filtered_contours:
                if cv.matchShapes(contour, saved_contour, cv.CONTOURS_MATCH_I1, 0.0) < error / 100:
                    cv.drawContours(grayscale_2_bgr, contour, -1, (0, 255, 127), 3)

        cv.imshow(window_name, grayscale_2_bgr)
        cv.imshow('Final', closing)

        key = cv.waitKey(30)
        if key == 27:
            break
        elif key == 99:
            actual_contours = filtered_contours
            if len(actual_contours) > 0:
                print('Captured!')
                saved_contour = get_max_contour(actual_contours)
                track = True
    cv.destroyAllWindows()


def filter_contours(contours, area_range):
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area_range[0] * 100000 < area < area_range[1] * 100000:
            filtered_contours.append(contour)
    return filtered_contours


def get_max_contour(contours):
    max_contour = contours[0]
    for contour in contours:
        if cv.contourArea(max_contour) < cv.contourArea(contour):
            max_contour = contour

    return max_contour


def on_trackbar_name(val):
    pass


detection()
