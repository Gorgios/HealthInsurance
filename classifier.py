import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as mpl


def generate_diff_chart(dtc, knn3, knn5, knn11, bayes):
    x_sticks = ["dtc", "knn3", "knn7", "knn11", "bayes"]
    x = list(range(0, len(x_sticks)))
    y = [dtc, knn3, knn5, knn11, bayes]
    mpl.xticks(x, x_sticks)
    mpl.bar(x, y, width=0.4, color="green", alpha=0.8)
    mpl.title("Porównanie poprawności klasyfikatorów")
    mpl.xlabel("Klasyfikator")
    mpl.ylabel("Poprawnosc")
    mpl.savefig('poprawnosc_klasyfikatorow.png')
    mpl.close()


class Classifier:
    train_inputs = []
    train_classes = []
    test_inputs = []
    test_classes = []

    def __init__(self, df, divide_ratio, class_column):
        c = df[class_column].values
        i = df.drop([class_column], axis=1)
        (self.train_inputs, self.test_inputs, self.train_classes, self.test_classes) = \
            train_test_split(i, c, train_size=divide_ratio)

    def dtc(self):
        dtc = DecisionTreeClassifier()
        dtc.fit(self.train_inputs, self.train_classes)
        decision_tree_score = dtc.score(self.test_inputs, self.test_classes)
        print("Poprawnosc dla drzewa decyzyjnego: " + str(decision_tree_score))
        plot_confusion_matrix(dtc, self.test_inputs, self.test_classes)
        mpl.title("Macierz błędu dla drzewa decyzyjnego")
        mpl.savefig('dtc_confusion_matrix.png')
        mpl.close()
        return decision_tree_score

    def knn(self, k):
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(self.train_inputs, self.train_classes)
        knn_score = knn.score(self.test_inputs, self.test_classes)
        print("Poprawnosc dla knn = {}: {}".format(k, knn_score))
        plot_confusion_matrix(knn, self.test_inputs, self.test_classes)
        mpl.title("Macierz błędu dla knn" + str(k))
        mpl.savefig('knn{}_confusion_matrix.png'.format(k))
        mpl.close()
        return knn_score

    def gnb(self):
        gnb = GaussianNB()
        gnb.fit(self.train_inputs, self.train_classes)
        gnb_score = gnb.score(self.test_inputs, self.test_classes)
        print("Poprawnosc dla naive Bayes : " + str(gnb_score))
        plot_confusion_matrix(gnb, self.test_inputs, self.test_classes)
        mpl.title("Macierz błędu dla naive Bayes")
        mpl.savefig('gnb_confusion_matrix.png')
        mpl.close()
        return gnb_score
