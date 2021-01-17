import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_db(ipt, out):
    df = pd.read_csv(ipt)
    labelEncoder = LabelEncoder()
    df['Gender'] = labelEncoder.fit_transform(df['Gender'])
    df['Vehicle_Age'] = labelEncoder.fit_transform(df['Vehicle_Age'])
    df['Vehicle_Damage'] = labelEncoder.fit_transform(df['Vehicle_Damage'])
    df.drop(['id'], axis=1, inplace=True)
    df.to_csv(out, index=False)
