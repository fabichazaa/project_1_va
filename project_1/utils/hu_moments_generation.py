import cv2 as cv
import csv
import glob
import numpy as np
import math


# writes Hu moments in a file
def write_hu_moments(label, writer):
    files = glob.glob('./shapes/' + label + '/*')
    # opens the folder in shapes with the name 'label' and gets all files
    hu_moments = []
    for file in files:
        hu_moments.append(hu_moments_of_file(file))
    for mom in hu_moments:
        flattened = mom.ravel()
        row = np.append(flattened, label)
        writer.writerow(row)


def generate_hu_moments_file():
    # generates new file
    with open('generated-files/shapes-hu-moments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        write_hu_moments('5-point-star', writer)
        write_hu_moments('rectangle', writer)
        write_hu_moments('triangle', writer)
        write_hu_moments('circle', writer)


def hu_moments_of_file(file):
    image = cv.imread(file)
    grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(grayscale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 67, 2)

    binary = 255 - binary
    # inverts
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    binary = cv.morphologyEx(binary, cv.MORPH_ERODE, kernel)
    contours, hierarchy = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv.contourArea)
    # gets max area contour

    moments = cv.moments(shape_contour)
    hu_moments = cv.HuMoments(moments)

    for i in range(0, 7):
        hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

    return hu_moments
