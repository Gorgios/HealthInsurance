import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix as pcm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as mpl
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, normalize


def generate_diff_chart(dtc, knn3, knn5, knn11, bayes, neural, forest):
    x_sticks = ["dtc", "knn3", "knn7", "knn11", "bayes", "neural", "randf"]
    x = list(range(0, len(x_sticks)))
    y = [dtc, knn3, knn5, knn11, bayes, neural, forest]
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
    scaler = MinMaxScaler()

    def __init__(self, df, divide_ratio, class_column):
        df.drop(['Gender', 'Driving_License'], axis=1)
        c = df[class_column].values.astype('int64')
        i = df.drop([class_column, 'Gender', 'Driving_License'], axis=1).values.astype('float64')
        randomsample = RandomOverSampler()
        i_new, c_new = randomsample.fit_sample(i, c)
        (self.train_inputs, self.test_inputs, self.train_classes, self.test_classes) = \
            train_test_split(i_new, c_new, train_size=divide_ratio, random_state=1)

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

    def neural(self):
        x_train = normalize(self.train_inputs)
        x_test = normalize(self.test_inputs)
        clf = MLPClassifier(solver='adam', activation='tanh', hidden_layer_sizes=(64, 2))
        clf.fit(x_train, self.train_classes)
        y_pred = clf.predict(x_test)
        score = accuracy_score(y_pred, self.test_classes)
        cm = confusion_matrix(y_pred, self.test_classes)
        print("Wynik dla sieci neuronowej", score)
        pcm(conf_mat=cm)
        mpl.title("Macierz błędu dla sieci neuronowej")
        mpl.savefig("neural_confusion_matrix.png")
        mpl.close()
        return score

    def random_forrest(self):
        randomforest = RandomForestClassifier()
        randomforest.fit(self.train_inputs, self.train_classes)
        y_pred = randomforest.predict(self.test_inputs)
        print("Srednia dla random forrest: ", accuracy_score(y_pred, self.test_classes))
        cm = confusion_matrix(y_pred, self.test_classes)
        pcm(conf_mat=cm)
        mpl.title("Macierz błędu dla random forrest")
        mpl.savefig("randomforest_confusion_matrix.png")
        mpl.close()
        return accuracy_score(y_pred, self.test_classes)
