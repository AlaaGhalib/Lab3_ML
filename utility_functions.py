import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, model_name="Model"):
    '''Evaluates the given model and visualizes its performance.'''
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\n--- {model_name} Performance ---\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Not Readmitted", "Readmitted"], 
                yticklabels=["Not Readmitted", "Readmitted"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()
    
    # ROC Curve and AUC Calculation (Only if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)

        # Handle Binary Classification
        if len(np.unique(y_test)) == 2:
            y_prob = y_prob[:, 1]  # Probability of class 1 (Readmitted)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot ROC Curve
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} - ROC Curve (Binary)")
            plt.legend(loc="lower right")
            plt.show()

            print(f"AUC Score: {roc_auc:.2f}")

        # Handle Multi-Class Classification
        else:
            y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
            plt.figure(figsize=(6, 4))
            for i, class_label in enumerate(np.unique(y_test)):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{model_name} - ROC Curve (Multiclass)")
            plt.legend()
            plt.show()