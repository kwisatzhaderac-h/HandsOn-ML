# %% Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

# %%
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8) # convert string type into int for algorithm
# %% Viewing an instance of the dataset

some_digit = X.iloc[0]
some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# %% Splitting into test and training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %% Binary Classifier - identifies two classes
# let's first try to classify the number 5 only
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %% Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %% Cross-validation of model
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# the method is not good for this case, as only 10% of the test set is 
# comprised of 5s. Data is skewed. Scoring by accuracy is not suitable.
'''
Using Stratified K Fold
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
'''
# %% Confusion Matrix
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)
# %% Precision and Recall - another way to evaluate performance of model
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

# when it predicts a 5, its only right 83% of the time
precision_score = precision_score(y_train_5, y_train_pred)
print(precision_score) 
# it is only able to detect 65% of the 5s in the training set
recall_score = recall_score(y_train_5, y_train_pred)
print(recall_score)
# combine precision and recall score to create f1 score metric
print(f1_score(y_train_5, y_train_pred))
print(2/(1/precision_score + 1/recall_score)) # precision score is harmonic mean. Will be the same as above.

# %% Findin the most optimum threshold for the decision function
from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                            method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# %%
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g', label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.grid(b=bool)
    plt.axis([-50000, 50000, 0, 1]) 

plt.figure(figsize=(8, 4))
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")                                             
plt.plot([threshold_90_precision], [recall_90_precision], "ro")                             
save_fig("precision_recall_vs_threshold_plot")                                              
plt.show()

# %%
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(b=bool)
    plt.axis([0, 1, 0, 1])

plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0, 0.9], 'r:')
plt.plot([0, recall_90_precision], [0.9, 0.9], 'r:')
plt.plot(recall_90_precision, 0.9, 'ro')
save_fig('precision_vs_recall_plot')
plt.show()
# %%
