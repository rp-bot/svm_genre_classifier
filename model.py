from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from _dataloader import clean_and_split_data, get_feature_vector_for_file
import numpy as np


def linear_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    # testing the linear classifier if it is capable?
    linear_classifier = svm.SVC(kernel='linear')
    linear_classifier.fit(X_train, y_train)
    y_hat_train = linear_classifier.predict(X_train)
    y_hat_test = linear_classifier.predict(X_test)

    train_precision = metrics.precision_score(
        y_train, y_hat_train, average='macro')
    test_precision = metrics.precision_score(
        y_test, y_hat_test, average='macro')

    train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average='macro')

    print(f"Training Precision: {
        train_precision:.4f}, Test Precision: {test_precision:.4f}")
    print(f"Training F1-score: {train_f1:.4f}, Test F1-score: {test_f1:.4f}")

    X_test_file = get_feature_vector_for_file(
        "128bpm_tr8_drm_id_006_0166.wav", 128)

    y_hat_train = linear_classifier.predict(np.array([X_test_file]))

    print(label_decoder.inverse_transform(y_hat_train))


def RBF_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    svm_rbf = svm.SVC(kernel='rbf', C=100, gamma='scale')
    # grid_search = GridSearchCV(
    #     svm_rbf, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    svm_rbf.fit(X_train, y_train)

    # best_svm = grid_search.best_estimator_

    y_pred = svm_rbf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    class_report = metrics.classification_report(y_test, y_pred)

    # print("Best Hyperparameters:", grid_search.best_params_)
    print("Test Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    X_test_file = get_feature_vector_for_file(
        "95bpm_tr8_drm_id_001_0069.wav", 95)

    y_test_file = svm_rbf.predict(np.array([X_test_file]))

    print(label_decoder.inverse_transform(y_test_file))


def random_forest_classifier():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_hat = rf_classifier.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_hat)
    precision = metrics.precision_score(
        y_test, y_hat, average='macro')  # Macro for multi-class
    recall = metrics.recall_score(
        y_test, y_hat, average='macro')  # Macro for multi-class
    # Macro for multi-class
    f1 = metrics.f1_score(y_test, y_hat, average='macro')
    conf_matrix = metrics.confusion_matrix(y_test, y_hat)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision (Macro): {precision:.2f}")
    print(f"Recall (Macro): {recall:.2f}")
    print(f"F1-score (Macro): {f1:.2f}")

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_hat))

    X_test_file = get_feature_vector_for_file(
        "data/out_of_ditribution/128bpm_tr8_drm_id_009_0281.wav", 125)

    y_test_file = rf_classifier.predict(np.array([X_test_file]))

    print(label_decoder.inverse_transform(y_test_file))


if __name__ == '__main__':

    # linear_SVM()

    # RBF_SVM()

    random_forest_classifier()
