import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as mpl
from sklearn.preprocessing import normalize



df = pd.read_csv("health_insurance.csv")
# Zmieniamy plec na liczby 1 = mezczyna 0 = kobieta
df.loc[df['Gender'] == 'Male', 'Gender'] = 1
df.loc[df['Gender'] == 'Female', 'Gender'] = 0
# zamieniamy wiek auta na liczby 0 = < 1 year, 1 = 1-2 years, 2 > 2 years
df.loc[df['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0
df.loc[df['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1
df.loc[df['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2
# zamieniamy uszkodzenie auta na liczby 1 = tak, 0 = nie
df.loc[df['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1
df.loc[df['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

all_classes = df['Response'].values
all_inputs = df.drop(['Response', 'id'], axis=1)
cn = ['tested_negative', 'tested_positive']
print(all_inputs)
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.67)
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
decision_tree_score = dtc.score(test_inputs, test_classes)
print("Poprawnosc dla drzewa decyzyjnego: " + str(decision_tree_score))
decision_tree_matrix = confusion_matrix(test_classes, dtc.predict(test_inputs))
print("Macierz błędu dla drzewa decyzyjnego: ")
print(decision_tree_matrix)

knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn5.fit(train_inputs, train_classes)
knn5_score = knn5.score(test_inputs, test_classes)
print("Poprawnosc dla knn = 5: " + str(knn5_score))
knn5_matrix = confusion_matrix(test_classes, knn5.predict(test_inputs))
print("Macierz bledu dla knn = 5")
print(knn5_matrix)

knn7 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn7.fit(train_inputs, train_classes)
knn7_score = knn7.score(test_inputs, test_classes)
print("Poprawnosc dla knn = 7: " + str(knn7_score))
knn7_matrix = confusion_matrix(test_classes, knn7.predict(test_inputs))
print("Macierz bledu dla knn = 7")
print(knn7_matrix)

knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn11.fit(train_inputs, train_classes)
knn11_score = knn11.score(test_inputs, test_classes)
print("Poprawnosc dla knn = 11: " + str(knn11_score))
knn11_matrix = confusion_matrix(test_classes, knn11.predict(test_inputs))
print("Macierz bledu dla knn = 11")
print(knn11_matrix)

gnb = GaussianNB()

gnb.fit(train_inputs, train_classes)
gnb_score = gnb.score(test_inputs, test_classes)
print("Poprawnosc dla naive Bays : " + str(gnb_score))
gnb_matrix = confusion_matrix(test_classes, gnb.predict(test_inputs))
print("Macierz bledu dla naive Bays")
print(gnb_matrix)

x_sticks = ["decision tree", "knn5", "knn7", "knn11", "bayes"]
x = list(range(0, len(x_sticks)))
y = [decision_tree_score, knn5_score, knn7_score, knn11_score, gnb_score]
mpl.xticks(x, x_sticks)
mpl.bar(x, y, width=0.4, color="green", alpha=0.8)
mpl.title("Porównanie poprawności klasyfikatorów")
mpl.xlabel("Klasyfikator")
mpl.ylabel("Poprawnosc")
mpl.show()
