import argparse
import os
import numpy as np
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from collections import defaultdict
from tqdm import tqdm

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return (np.trace(C)/np.sum(C))

def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[i, i] / np.sum(C[i, :]) for i in range(C.shape[0])]

def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[i, i] / np.sum(C[:, i]) for i in range(C.shape[0])]

# These are the classifiers for 3.2, 3.3, and 3.4
classifier_trainings = {
    "SGDClassifier" : SGDClassifier(),
    "GaussianNB" : GaussianNB(),
    "RandomForestClassifier" : RandomForestClassifier(max_depth=5, n_estimators=10),
    "MLPClassifier" : MLPClassifier(alpha=0.05),
    "AdaBoostClassifier" : AdaBoostClassifier()
}

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    classifier_list = list(classifier_trainings.keys())
    best_classifier_accuracy= 0

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for classifier_name in tqdm(classifier_list):
            conf_matrix = confusion_matrix(y_test, (classifier_trainings[classifier_name].fit(X_train, y_train)).predict(X_test))
            classifier_accuracy = accuracy(conf_matrix)
            if classifier_accuracy > best_classifier_accuracy:
                best_classifier_accuracy = classifier_accuracy
                iBest = classifier_list.index(classifier_name)
            # For each classifier, compute results and write the following output:
                outf.write(f'Results for {classifier_name}:\n')  # Classifier name
                outf.write(f'\tAccuracy: {classifier_accuracy:.4f}\n')
                outf.write(f'\tRecall: {[round(item, 4) for item in recall(conf_matrix)]}\n')
                outf.write(f'\tPrecision: {[round(item, 4) for item in precision(conf_matrix)]}\n')
                outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return iBest

def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    classifier_list = list(classifier_trainings.keys())
    best_classifier = classifier_trainings[classifier_list[iBest]]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))
        for num_train in tqdm([1000, 5000, 10000, 15000, 20000]):
            if num_train == 1000:
                X_1k, y_1k = X_train[0:1000, :], y_train[0:1000]
            conf_matrix = confusion_matrix(y_test, best_classifier.fit(X_train[0:num_train, :], y_train[0:num_train]).predict(X_test))
            outf.write(f'{num_train}: {accuracy(conf_matrix):.4f}\n')
        outf.write(f'There is a noticeable trend such that as the number of data trained on the classifier results in greater accuracy. This is generally expected.')

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    classifier_list = list(classifier_trainings.keys())
    best_classifier = classifier_trainings[classifier_list[i]]

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        for k_feat in tqdm([5, 50]):
            best_k = SelectKBest(f_classif, k_feat)
            if k_feat == 5:
                X_1k_new = best_k.fit_transform(X_1k, y_1k)
                X_1k_new_test = best_k.transform(X_test)
                features = best_k.get_support(indices=True).tolist()
                X_32k_new = best_k.fit_transform(X_train, y_train)
                X_32k_new_test = best_k.transform(X_test)
                top_5 = best_k.get_support(indices=True).tolist()
            else:
                best_k.fit(X_train, y_train)
            # for each number of features k_feat, write the p-values for
            # that number of features:
            p_values = best_k.pvalues_
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        accuracy_1k = accuracy(confusion_matrix(y_test, best_classifier.fit(X_1k_new, y_1k).predict(X_1k_new_test)))
        accuracy_full = accuracy(confusion_matrix(y_test, best_classifier.fit(X_32k_new, y_train).predict(X_32k_new_test)))

        feature_intersection = list(set(features) & set(top_5))
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')
        outf.write(f'a) 136: receptiviti_family_oriented, 83: liwc_motion, 96: liwc_relativ\n This is later answered in part c) as well. \n') #This may change
        outf.write(f'b) p values are the same in these results. In fact, they do not change for any chosen k value.\n')
        outf.write(f'c) 136: receptiviti_family_oriented, 83: liwc_motion, 119: receptiviti_ambitious, 96: liwc_relativ, 153: receptiviti_money_oriented\n')
        outf.write(f'Here, it we may see that features that are associated with LIWC family oriented, which is plausible, since family invokes emotional feelings moreso than objective reason. \n Features associated with ambition may be politically charged as they may be corresponding with political movements, etc. \n Motion on the other hand may also be political since, as with ambitiousness, it may be corresponding with political movements.\n Money is also an indicator for political bias since topics on these subreddits may speak about certain affordances and inaffordances in life, as well as the economy, social welfare, struggles of poverty, etc.')

def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    total_X, total_y =  np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
    classifier_list = list(classifier_trainings.keys())
    kfold = KFold(n_splits=5, shuffle=True)
    kfolds = defaultdict(list)
    kfold_accuracies = []
    p_values = []
    best_acc = 0

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        for classifier_name in tqdm(classifier_list):
            acc = 0
            for train_idx, test_idx in kfold.split(total_X, total_y):
                conf_matrix = confusion_matrix(total_y[test_idx],
                classifier_trainings[classifier_name].fit(total_X[train_idx],
                total_y[train_idx]).predict(total_X[test_idx]))
                kfolds[classifier_name].append(accuracy(conf_matrix))
            mean_acc = (sum(kfolds[classifier_name]) / 5)
            kfold_accuracies.append(mean_acc)
            if mean_acc > best_acc:
                best_class = classifier_name
                best_acc = mean_acc

        best = kfolds[best_class]
        for classifier_name in set(classifier_list) - {best_class}:
            p_values.append(ttest_rel(best, kfolds[classifier_name]).pvalue)

        outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    # TODO: load data and split into train and test.
    # TODO : complete each classification experiment, in sequence.
    feats = np.load(args.input)['arr_0']
    X, y = feats[:, :-1], feats[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for i in tqdm(range(4)):
        if i == 0:
            iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
        elif i == 1:
            (X_1k, y_1k) = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
        elif i == 2:
            class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
        else:
            class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
