import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle

class_names = ['mountain', 'street', 'glacier']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 100  # Adjust this based on your server's memory capacity

def process_images(dataset, folder, start, end):
    images = []
    labels = []
    files = os.listdir(os.path.join(dataset, folder))[start:end]
    for file in files:
        img_path = os.path.join(os.path.join(dataset, folder), file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)
        labels.append(class_names_label[folder])
    return images, labels

def load_data():
    datasets = ['seg_train', 'seg_test']
    output = []

    for dataset in datasets:
        dataset_images = []
        dataset_labels = []

        for folder in os.listdir(dataset):
            folder_path = os.path.join(dataset, folder)
            num_images = len(os.listdir(folder_path))
            for i in tqdm(range(0, num_images, BATCH_SIZE)):
                images, labels = process_images(dataset, folder, i, min(i + BATCH_SIZE, num_images))
                dataset_images.extend(images)
                dataset_labels.extend(labels)

        dataset_images = np.array(dataset_images, dtype='float32') / 255.0
        dataset_labels = np.array(dataset_labels, dtype='int32')

        output.append((dataset_images, dataset_labels))

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

