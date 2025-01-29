from sklearn import svm
from sklearn import metrics
from _dataloader import clean_and_split_data

def linear_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    # testing the linear classifier if it is capable?
    linear_classifier = svm.SVC(kernel='linear')
    linear_classifier.fit(X_train, y_train)
    y_hat_train = linear_classifier.predict(X_train)
    y_hat_test = linear_classifier.predict(X_test)

    train_precision = metrics.precision_score(y_train, y_hat_train, average='macro')
    test_precision = metrics.precision_score(y_test, y_hat_test, average='macro')

    train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average='macro')

    print(f"Training Precision: {
        train_precision:.4f}, Test Precision: {test_precision:.4f}")
    print(f"Training F1-score: {train_f1:.4f}, Test F1-score: {test_f1:.4f}")


def RBF_SVM():
    pass