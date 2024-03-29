import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import os
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle

C_options = [2 ** i for i in range(-2, 2, 1)]
gamma_options = [2 ** i for i in range(-2, 2, 1)]

# Adjust ranges based on granularity
kernel_options = ['linear', 'rbf']

param_grid = {
    'C': C_options,
    'kernel': kernel_options,
    'gamma': gamma_options}



class_names = ['mountain', 'street', 'glacier']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
IMAGE_SIZE = (150, 150)

def load_data():
    path1 = os.path.join("seg_train")
    path2 = os.path.join("seg_test")
    datasets = [path1, path2]
    images = []
    labels = []

    for dataset in datasets:
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

    return images, labels

def get_processed_data():
    images, labels = load_data()
    images, labels = shuffle(images, labels, random_state=25)

    # Normalize the images
    images = images / 255.0

    # Reshape the images for SVM
    images = images.reshape(-1, 150*150*3)

    return {'data': images, 'target': labels}

# Use this function to get your dataset
data = get_processed_data()

X, y = data['data'], data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get parameter grid based on user-defined granularity
print("Parameter Grid:", param_grid)

# Determine CV folds
cv_folds = 4 if X_train.shape[0] < 500 else 2
print("CV Folds:", cv_folds)

# Initialize the SVM classifier
model = SVC(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_folds, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
print(classification_report(y_test, y_pred))
