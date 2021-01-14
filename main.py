import prepare_database as prep
import statistics as stat
import classifier as c
import pandas as pd

output = "prepared.csv"
prep.prepare_db("health_insurance.csv", output)
df = pd.read_csv(output)
stat.get_statistics(df)
classifier = c.Classifier(df, 0.67, 'Response')
dtc = classifier.dtc()
knn3 = classifier.knn(3)
knn5 = classifier.knn(5)
knn11 = classifier.knn(11)
bayes = classifier.gnb()
c.generate_diff_chart(dtc,knn3,knn5,knn11,bayes)
