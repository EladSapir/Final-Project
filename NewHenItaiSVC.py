import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt

class_names = ['mountain', 'street', 'glacier']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (150, 150)

def load_data():
    datasets = ['seg_train', 'seg_test']
    output = []

    for dataset in datasets:
        images = []
        labels = []

        print("Loading {}".format(dataset))

        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                img_path = os.path.join(os.path.join(dataset, folder), file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output

(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 150*150*3)
test_images = test_images.reshape(-1, 150*150*3)

# Parameters to iterate over
C_values = [1, 5]
degrees = [2, 3]

for C in C_values:
    for degree in degrees:
        print(f"Training SVM with C={C}, degree={degree}, kernel=poly")

        clf = svm.SVC(C=C, kernel='poly', degree=degree)
        clf.fit(train_images, train_labels)

        predictions = clf.predict(test_images)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Accuracy with C={C}, degree={degree}, kernel=poly: {accuracy}")

        print(classification_report(test_labels, predictions, target_names=class_names))
