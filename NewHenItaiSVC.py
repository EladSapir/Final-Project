import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


#class_names = ['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_names = ['mountain','street', 'glacier']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (150, 150)


def load_data():
    """
        Load the data:
            - 14,034 images to train the network.
            - 3,000 images to evaluate how accurately the network learned to classify images.
    """

    datasets = ['seg_train', 'seg_test']
    #datasets = ['C:/Users/hanig/Desktop/seg_train', 'C:/Users/hanig/Desktop/seg_test']
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output




(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))



_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts,
                    'test': test_counts},
             index=class_names
            ).plot.bar()
#plt.show()



train_images = train_images / 255.0
test_images = test_images / 255.0






#display_examples(class_names, train_images, train_labels)


# Create an SVM model
train_images = train_images.reshape(-1, 150*150*3)
test_images = test_images.reshape(-1, 150*150*3)



# Define the parameter grid
param_grid = {'C': [1,3,5,7,8,10],'cache_size':[3000],'gamma':['auto','scale'], 'kernel': ['rbf']}
#param_grid = {'C': [1,5,8,10],'cache_size':[3000],'degree':['2','3'], 'kernel': ['poly']}
# Create a base model
svc = svm.SVC()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid,
                           cv=2, n_jobs=-1, verbose=2)

print("begin training with grid search")
# Fit the grid search to the data
grid_search.fit(train_images, train_labels)

# Use the best parameters for the model
clf = grid_search.best_estimator_

# You can print out the best parameters like this:
print(f"Best parameters found: {grid_search.best_params_}")


# Test the model on the test data

predictions = clf.predict(test_images)

# Print the classification report
print(classification_report(test_labels, predictions, target_names=class_names))

# Print the accuracy
print("Accuracy:", accuracy_score(test_labels, predictions))



accuracy = clf.score(test_images, test_labels)
print("score with new data: ",accuracy)
plt.plot(accuracy)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy')
plt.legend()
plt.show()

CM = confusion_matrix(test_labels, predictions)
