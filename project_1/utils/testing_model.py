import cv2 as cv
import numpy as np
import glob

from Project.project_1.utils.hu_moments_generation import hu_moments_of_file
from Project.project_1.utils.label_converters import int_to_label


def load_and_test(model):
    files = glob.glob('../shapes/testing/*')
    for f in files:
        hu_moments = hu_moments_of_file(f)
        sample = np.array([hu_moments], dtype=np.float32)
        test_response = model.predict(sample)[1]

        image = cv.imread(f)
        image_with_text = cv.putText(image, int_to_label(test_response), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                                     (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Result", image_with_text)
        cv.waitKey(0)
