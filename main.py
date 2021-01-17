import prepare_database as prep
import statistics as stat
import classifier as c
import pandas as pd

output = "prepared.csv"
prep.prepare_db("health_insurance.csv", output)
df = pd.read_csv(output)
stat.get_statistics(df)
classifier = c.Classifier(df, 0.7, 'Response')
knn3 = classifier.knn(3)
knn5 = classifier.knn(5)
knn11 = classifier.knn(11)
dtc = classifier.dtc()
bayes = classifier.gnb()
neural = classifier.neural()
forest = classifier.random_forrest()
c.generate_diff_chart(dtc, knn3, knn5, knn11, bayes, neural, forest)
