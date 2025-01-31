from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from _dataloader import clean_and_split_data, get_feature_vector_for_file
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(train_metrics, test_metrics, metric_names, model_name, plot_name):
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, train_metrics, width, label='Train')
    bars2 = ax.bar(x + width/2, test_metrics, width, label='Test')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig(f"plots/{plot_name}.png")


def plot_confusion_matrix(y_true, y_pred, class_labels, plot_name, model_name):
    cm = metrics.confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"plots/{plot_name}.png")


def linear_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    # testing the linear classifier if it is capable?
    linear_classifier = svm.SVC(kernel='linear')
    linear_classifier.fit(X_train, y_train)
    y_hat_train = linear_classifier.predict(X_train)
    y_hat_test = linear_classifier.predict(X_test)

    train_accuracy = metrics.accuracy_score(y_train, y_hat_train)
    test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

    train_precision = metrics.precision_score(
        y_train, y_hat_train, average='macro')
    test_precision = metrics.precision_score(
        y_test, y_hat_test, average='macro')

    train_recall = metrics.recall_score(y_train, y_hat_train, average='macro')
    test_recall = metrics.recall_score(y_test, y_hat_test, average='macro')

    train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average='macro')

    print("Classification Report:")
    print(metrics.classification_report(
        y_test, y_hat_test, target_names=label_decoder.classes_))

    plot_confusion_matrix(y_test, y_hat_test, label_decoder.classes_,
                          plot_name="linear_svm_confusion_matrix", model_name="Linear SVM")

    train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(train_metrics, test_metrics, metric_names,
                 plot_name="linear_svm_metrics", model_name="Linear SVM")

    # X_test_file = get_feature_vector_for_file(
    #     "data/out_of_ditribution/95bpm_tr8_drm_id_001_0069.wav", 128)

    # y_hat_train = linear_classifier.predict(np.array([X_test_file]))

    # print(label_decoder.inverse_transform(y_hat_train))


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

    y_hat_train = svm_rbf.predict(X_train)
    y_hat_test = svm_rbf.predict(X_test)

    train_accuracy = metrics.accuracy_score(y_train, y_hat_train)
    test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

    train_precision = metrics.precision_score(
        y_train, y_hat_train, average='macro')
    test_precision = metrics.precision_score(
        y_test, y_hat_test, average='macro')

    train_recall = metrics.recall_score(y_train, y_hat_train, average='macro')
    test_recall = metrics.recall_score(y_test, y_hat_test, average='macro')

    train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average='macro')

    # Print Classification Report
    print("Classification Report:\n", metrics.classification_report(
        y_test, y_hat_test, target_names=label_decoder.classes_))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_hat_test, label_decoder.classes_,
                          plot_name="RBF_svm_confusion_matrix", model_name="RBF SVM")

    # Plot Accuracy, Precision, Recall, and F1-score
    train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(train_metrics, test_metrics, metric_names,
                 plot_name="RBF_svm_metrics", model_name="RBF SVM")

    # X_test_file = get_feature_vector_for_file(
    #     "data/out_of_ditribution/95bpm_tr8_drm_id_001_0069.wav", 95)

    # y_test_file = svm_rbf.predict(np.array([X_test_file]))

    # print(label_decoder.inverse_transform(y_test_file))


def random_forest_classifier():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_hat_train = rf_classifier.predict(X_train)
    y_hat_test = rf_classifier.predict(X_test)

    # Compute Metrics
    train_accuracy = metrics.accuracy_score(y_train, y_hat_train)
    test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

    train_precision = metrics.precision_score(
        y_train, y_hat_train, average='macro')
    test_precision = metrics.precision_score(
        y_test, y_hat_test, average='macro')

    train_recall = metrics.recall_score(y_train, y_hat_train, average='macro')
    test_recall = metrics.recall_score(y_test, y_hat_test, average='macro')

    train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average='macro')

    # Print Classification Report
    print("Classification Report:\n", metrics.classification_report(
        y_test, y_hat_test, target_names=label_decoder.classes_))

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_hat_test, label_decoder.classes_,
                          plot_name="RF_confusion_matrix", model_name="RF")

    # Plot Accuracy, Precision, Recall, and F1-score
    train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(train_metrics, test_metrics, metric_names,
                 plot_name="RF_metrics", model_name="RF")

    X_test_file = get_feature_vector_for_file(
        "data/out_of_ditribution/120bpm_hse_drm_id_001_0018.wav", 0)

    y_test_file_proba = rf_classifier.predict_proba(np.array([X_test_file]))

    class_names = label_decoder.classes_

    for class_name, prob in zip(class_names, y_test_file_proba[0]):

        print(f"{class_name}: {prob:.4f}")


if __name__ == '__main__':

    # linear_SVM()

    # RBF_SVM()

    random_forest_classifier()
