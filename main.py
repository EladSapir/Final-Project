from sklearn.metrics import confusion_matrix

from EladTomerSolalSVC import *
from ToolsKit import *

def main():
    # improveOurPreviousProject()
    data=(UseToolKit([1,1,1,0,0],'Database.csv','NA',1))[-1][0]





    X = data.iloc[:, :-1]  # Select all columns except the last one as features
    y = data.iloc[:, -1]  # Select the last column as label

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the SVC
    svc_model = SVC(kernel='rbf', gamma=1, C=10)

    # Train the model
    svc_model.fit(X_train, y_train)

    # Make predictions
    predictions = svc_model.predict(X_test)

    # Evaluate the model
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

if __name__== '__main__':
    main()