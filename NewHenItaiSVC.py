import numpy as np
import os
import cv2
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt

class_names = ['mountain', 'street', 'glacier']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (150, 150)

def process_folder(data):
    dataset, folder = data
    label = class_names_label[folder]
    images = []
    labels = []

    for file in os.listdir(os.path.join(dataset, folder)):
        img_path = os.path.join(os.path.join(dataset, folder), file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        images.append(image)
        labels.append(label)

    return images, labels

def load_data():
    datasets = ['seg_train', 'seg_test']
    output = []

    for dataset in datasets:
        folder_paths = [(dataset, folder) for folder in os.listdir(dataset)]

        with Pool(os.cpu_count()) as p:
            results = p.map(process_folder, folder_paths)

        # Combining results from all folders
        images, labels = zip(*results)
        images = [img for sublist in images for img in sublist]
        labels = [label for sublist in labels for label in sublist]

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

# Load data
(train_images, train_labels), (test_images, test_labels) = load_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 150*150*3)
test_images = test_images.reshape(-1, 150*150*3)

# Parameters to iterate over
clf = svm.SVC(C=10, kernel='poly', degree=3 ,cache_size=8000)
clf.fit(train_images, train_labels)
predictions = clf.predict(test_images)
accuracy = accuracy_score(test_labels, predictions)

