from pprint import pprint
from sklearn import svm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# from sklearn.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn import metrics
from _dataloader import clean_and_split_data, get_feature_vector_for_file
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm


def plot_metrics(test_metrics, metric_names, model_name, plot_name):
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars2 = ax.bar(x + width / 2, test_metrics, width, label="Test")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.savefig(f"plots/{plot_name}.png")


def plot_confusion_matrix(y_true, y_pred, class_labels, plot_name, model_name):
    cm = metrics.confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"plots/{plot_name}.png")


def plot_feature_vector_distribution(X_train_scaled, y_train):

    feature_names = [
        "mean_onset",
        "std_onset",
        "mean_zcr",
        "std_zcr",
        "mean_ioi",
        "std_ioi",
        "tempo",
        "mean_centroid",
        "std_centroid",
        "mean_bandwidth",
        "std_bandwidth",
        "mean_rolloff",
        "std_rolloff",
        "mean_flatness",
        "std_flatness",
    ]

    feature_names += [f"MFCC_{idx+1}_mean" for idx in range(13)]

    feature_names += [f"MFCC_{idx+1}_std" for idx in range(13)]

    class_labels = ["samba", "hip_hop", "pop_rock", "tr_909"]

    # Convert normalized features to a DataFrame
    df = pd.DataFrame(X_train_scaled, columns=feature_names)
    df["Class"] = [class_labels[label] for label in y_train]

    num_features = len(feature_names)
    num_cols = 10
    num_rows = int(np.ceil(num_features / num_cols))

    plt.figure(figsize=(40, num_rows * 4))

    for i, feature in tqdm(enumerate(feature_names), total=num_features):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.kdeplot(
            data=df, x=feature, hue="Class", common_norm=False, fill=True, alpha=0.4
        )
        plt.title(feature)
        plt.xlabel("")
        plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig("plots/feature_vector_distributions_2.png")


def plot_sfs_progression(estimator, X_train_scaled, y_train):
    """
    Plots accuracy progression by iterating through feature subsets manually.

    Parameters:
    - estimator: Model used for feature selection (e.g., SVM).
    - X_train_scaled: Scaled training data.
    - y_train: Training labels.
    """
    num_features = []
    accuracy_scores = []

    # Ensure n_features_to_select is within valid range
    max_features = min(X_train_scaled.shape[1] // 2, X_train_scaled.shape[1] - 1)

    for i in tqdm(range(1, max_features + 1)):  # Limit to safe number of features
        sfs_temp = SFS(
            estimator,
            n_features_to_select=i,
            direction="forward",
            cv=5,
            scoring="accuracy",
        )
        sfs_temp.fit(X_train_scaled, y_train)
        selected_X = sfs_temp.transform(X_train_scaled)

        # Perform cross-validation to get accuracy
        scores = cross_val_score(
            estimator, selected_X, y_train, cv=5, scoring="accuracy"
        )
        num_features.append(i)
        accuracy_scores.append(scores.mean())

    plt.figure(figsize=(8, 6))
    plt.plot(num_features, accuracy_scores, marker="o", linestyle="-", color="b")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Cross-validated Accuracy")
    plt.title("SFS Feature Selection Progress")
    plt.grid(True)
    plt.xticks(num_features)
    plt.savefig("plots/SFS_progression.png")


def linear_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"number of total features: {len(X_train_scaled[0])}")
    baseline_model = svm.SVC(kernel="linear")
    baseline_model.fit(X_train_scaled, y_train)
    y_hat_baseline = baseline_model.predict(X_test_scaled)

    baseline_accuracy = metrics.accuracy_score(y_test, y_hat_baseline)
    baseline_precision = metrics.precision_score(
        y_test, y_hat_baseline, average="macro"
    )
    baseline_recall = metrics.recall_score(y_test, y_hat_baseline, average="macro")
    baseline_f1 = metrics.f1_score(y_test, y_hat_baseline, average="macro")

    print("\nClassification Report:")
    print(
        metrics.classification_report(
            y_test, y_hat_baseline, target_names=label_decoder.classes_
        )
    )
    # Plot Metrics
    test_metrics = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(
        test_metrics,
        metric_names,
        plot_name="baseline_linear_svm_metrics",
        model_name="Baseline Linear SVM",
    )

    # Plot Confusion Matrix for Regular Validation
    plot_confusion_matrix(
        y_test,
        y_hat_baseline,
        label_decoder.classes_,
        plot_name="baseline_linear_svm_confusion_matrix",
        model_name="Baseline Linear SVM",
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_classifier = svm.SVC(kernel="linear")

    # Plot the SFS accuracy progression
    plot_sfs_progression(base_classifier, X_train_scaled, y_train)

    # Perform final feature selection
    sfs_final = SFS(
        base_classifier,
        n_features_to_select="auto",
        direction="forward",
        cv=5,
        scoring="accuracy",
    )
    sfs_final.fit(X_train_scaled, y_train)

    # Get the best feature indices
    best_feature_indices = sfs_final.get_support(indices=True)
    X_train_selected = sfs_final.transform(X_train_scaled)
    X_test_selected = sfs_final.transform(X_test_scaled)

    print(f"Best Feature Indices: {best_feature_indices}")
    print(f"Number of Best Features: {len(best_feature_indices)}")

    # Train final model with selected features
    final_model = svm.SVC(kernel="linear")
    final_model.fit(X_train_selected, y_train)

    y_hat_sfs = final_model.predict(X_test_selected)

    sfs_accuracy = metrics.accuracy_score(y_test, y_hat_sfs)
    sfs_precision = metrics.precision_score(y_test, y_hat_sfs, average="macro")
    sfs_recall = metrics.recall_score(y_test, y_hat_sfs, average="macro")
    sfs_f1 = metrics.f1_score(y_test, y_hat_sfs, average="macro")

    print("\nSFS Classification Report:")
    print(
        metrics.classification_report(
            y_test, y_hat_sfs, target_names=label_decoder.classes_
        )
    )

    # Plot SFS model metrics
    test_metrics = [sfs_accuracy, sfs_precision, sfs_recall, sfs_f1]
    plot_metrics(
        test_metrics,
        metric_names,
        plot_name="sfs_linear_svm_metrics",
        model_name="SFS Linear SVM",
    )

    # Plot Confusion Matrix for SFS-selected Model
    plot_confusion_matrix(
        y_test,
        y_hat_sfs,
        label_decoder.classes_,
        plot_name="sfs_linear_svm_confusion_matrix",
        model_name="SFS Linear SVM",
    )


def RBF_SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    }

    svm_rbf = svm.SVC(kernel="rbf", C=100, gamma="scale")
    # grid_search = GridSearchCV(
    #     svm_rbf, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    svm_rbf.fit(X_train, y_train)

    y_hat_test = svm_rbf.predict(X_test)

    test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

    test_precision = metrics.precision_score(y_test, y_hat_test, average="macro")

    test_recall = metrics.recall_score(y_test, y_hat_test, average="macro")

    test_f1 = metrics.f1_score(y_test, y_hat_test, average="macro")

    # Print Classification Report
    print(
        "Classification Report:\n",
        metrics.classification_report(
            y_test, y_hat_test, target_names=label_decoder.classes_
        ),
    )

    # Plot Confusion Matrix
    plot_confusion_matrix(
        y_test,
        y_hat_test,
        label_decoder.classes_,
        plot_name="RBF_svm_confusion_matrix",
        model_name="RBF SVM",
    )

    # Plot Accuracy, Precision, Recall, and F1-score

    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(
        test_metrics, metric_names, plot_name="RBF_svm_metrics", model_name="RBF SVM"
    )

    # X_test_file = get_feature_vector_for_file(
    #     "data/out_of_ditribution/95bpm_tr8_drm_id_001_0069.wav", 95)

    # y_test_file = svm_rbf.predict(np.array([X_test_file]))

    # print(label_decoder.inverse_transform(y_test_file))


def random_forest_classifier():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # y_hat_train = rf_classifier.predict(X_train)
    y_hat_test = rf_classifier.predict(X_test)

    # Compute Metrics
    # train_accuracy = metrics.accuracy_score(y_train, y_hat_train)
    test_accuracy = metrics.accuracy_score(y_test, y_hat_test)

    test_precision = metrics.precision_score(y_test, y_hat_test, average="macro")

    # train_recall = metrics.recall_score(y_train, y_hat_train, average='macro')
    test_recall = metrics.recall_score(y_test, y_hat_test, average="macro")

    # train_f1 = metrics.f1_score(y_train, y_hat_train, average='macro')
    test_f1 = metrics.f1_score(y_test, y_hat_test, average="macro")

    # Print Classification Report
    print(
        "Classification Report:\n",
        metrics.classification_report(
            y_test, y_hat_test, target_names=label_decoder.classes_
        ),
    )

    # Plot Confusion Matrix
    plot_confusion_matrix(
        y_test,
        y_hat_test,
        label_decoder.classes_,
        plot_name="RF_confusion_matrix",
        model_name="RF",
    )

    test_metrics = [test_accuracy, test_precision, test_recall, test_f1]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]

    plot_metrics(test_metrics, metric_names, plot_name="RF_metrics", model_name="RF")

    X_test_file = get_feature_vector_for_file(
        "data/out_of_ditribution/recording-1-30-2025,-7-22-17-PM.wav", 0
    )

    y_test_file_proba = rf_classifier.predict_proba(np.array([X_test_file]))

    class_names = label_decoder.classes_

    for class_name, prob in zip(class_names, y_test_file_proba[0]):

        print(f"{class_name}: {prob:.4f}")


def SVM():
    X_train, X_test, y_train, y_test, label_decoder = clean_and_split_data()

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # plot_feature_vector_distribution(X_train_scaled, y_train)
    X_train_experiment_1 = X_train_scaled[:, :7]
    X_test_experiment_1 = X_test_scaled[:, :7]
    print(X_train_experiment_1.shape)

    X_train_experiment_2 = X_train_scaled[:, 8:]
    X_test_experiment_2 = X_test_scaled[:, 8:]
    print(X_train_experiment_2.shape)

    # ================================================================== #
    # Experiemnt 1 only time domain features
    # Fit the data and predict (base model)
    experiment_1_base = svm.SVC(kernel="linear")
    experiment_1_base.fit(X_train_experiment_1, y_train)
    y_hat_experiment_1_base = experiment_1_base.predict(X_test_experiment_1)

    experiment_1 = SFS(
        experiment_1_base,
        k_features=6,
        forward=True,
        floating=False,
        verbose=2,
        scoring="accuracy",
        cv=5,
    )
    experiment_1.fit(X_train_experiment_1, y_train)
    y_hat_experiment_1 = experiment_1.predict(X_test_experiment_1)

    fig1 = plot_sfs(experiment_1.get_metric_dict(), kind="std_dev")

    # plt.ylim([0.8, 1])
    plt.title("Sequential Forward Selection (w. StdDev)")
    plt.grid()
    plt.savefig("temp.png")

    # compute metrics
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    experiment_1_base_accuracy = metrics.accuracy_score(y_test, y_hat_experiment_1_base)
    experiment_1_accuracy = metrics.accuracy_score(y_test, y_hat_experiment_1)
    experiment_1__base_precision = metrics.precision_score(
        y_test, y_hat_experiment_1_base, average="macro"
    )
    experiment_1_precision = metrics.precision_score(
        y_test, y_hat_experiment_1, average="macro"
    )
    experiment_1_recall = metrics.recall_score(
        y_test, y_hat_experiment_1, average="macro"
    )
    experiment_1_f1 = metrics.f1_score(y_test, y_hat_experiment_1, average="macro")

    print("\nClassification Report:")
    print(
        metrics.classification_report(
            y_test, y_hat_experiment_1, target_names=label_decoder.classes_
        )
    )
    # Plot Metrics
    experiment_1_test_metrics = [
        experiment_1_accuracy,
        experiment_1_precision,
        experiment_1_recall,
        experiment_1_f1,
    ]

    plot_metrics(
        experiment_1_test_metrics,
        metric_names,
        plot_name="experiment_1_linear_svm_metrics",
        model_name="experiment_1 Linear SVM",
    )

    # Plot Confusion Matrix for Regular Validation
    plot_confusion_matrix(
        y_test,
        y_hat_experiment_1,
        label_decoder.classes_,
        plot_name="experiment_1_linear_svm_confusion_matrix",
        model_name="experiment_1 Linear SVM",
    )
    # ================================================================== #
    # experiment_2 =


if __name__ == "__main__":

    # linear_SVM()
    SVM()
    # RBF_SVM()

    # random_forest_classifier()
