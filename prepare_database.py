import pandas as pd


def prepare_db(ipt, out):
    df = pd.read_csv(ipt)
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
    df.drop(['id', 'Region_Code', 'Policy_Sales_Channel'], axis=1, inplace=True)
    df.to_csv(out, index=False)
