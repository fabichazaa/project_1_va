import cv2 as cv
import csv
import numpy as np

from Project.project_1.utils.label_converters import label_to_int

trainData = []
trainLabels = []


def load_training_set():
    global trainData
    # variable that can be used outside the function
    global trainLabels

    # CSV: comma-separated-values
    # when we call on the main generate-hu-moments, we then can train the data we got
    with open('generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()
            floats = []
            for n in row:
                floats.append(float(n))
                # casts hu moments -> float
            trainData.append(np.array(floats, dtype=np.float32))
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32))
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)


def train_model():
    # tree is the model that receives the features and returns a label
    load_training_set()

    tree = cv.ml.DTrees_create()
    tree.setCVFolds(1)
    tree.setMaxDepth(10)
    tree.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)
    return tree

