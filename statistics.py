import pandas as pd
import matplotlib.pyplot as mpl


def get_statistics(df):
    print("Ilość rekordów w bazie " + str(len(df)))
    print()

    genders = df['Gender']
    print("Srednia występowania plci: " + str(genders.mean()))
    print("Występowanie płci 1 - mezczyzna 0 - kobieta: ")
    count_genders = genders.value_counts()
    print(count_genders)
    print()

    ages = df['Age']
    print("Srednia wieku " + str(ages.mean()))
    print("Najstarsza osoba " + str(ages.max()))
    print("Najmlodsza osoba " + str(ages.min()))
    count_ages = ages.value_counts()
    print("Najczesciej wystepujacy wiek " + str(count_ages.idxmax()))
    # Wykres
    count_ages.sort_index().plot()
    mpl.title("Ilość klientów a ich wiek")
    mpl.xlabel("Wiek")
    mpl.ylabel("Ilość klientów")
    mpl.savefig("klient_a_wiek.png")
    mpl.close()
    print()

    driving_licences = df['Driving_License']
    print("Srednia posiadania prawa jazdy " + str(driving_licences.mean()))
    count_driving_licences = driving_licences.value_counts()
    print("Posiadanie prawa jazdy 1 - tak, 0 - nie")
    print(count_driving_licences)
    print()

    previously_insured = df['Previously_Insured']
    print("Srednia wczesniej ubezpieczonych pojazdow " + str(previously_insured.mean()))
    count_previously_insured = previously_insured.value_counts()
    print("Poprzednio ubezpieczeni 1 - tak, 0 - nie")
    print(count_previously_insured)
    print()

    vehicle_ages = df['Vehicle_Age']
    count_vehicle_ages = vehicle_ages.value_counts()
    print("Wiek pojazdu : 0 - < 1 rok, 1 - 1-2 lata, 2 - > 2 lata")
    print(count_vehicle_ages)
    # Wykres
    count_vehicle_ages.sort_index().plot(kind="bar", figsize=(7, 6))
    mpl.xticks(count_vehicle_ages.sort_index().keys(), [' < 1 rok', '1-2 lat', '> 2 lata'], rotation=45)
    mpl.title("Ilość klientów a wiek auta")
    mpl.xlabel("Wiek auta")
    mpl.ylabel("Ilosc klientow")
    mpl.savefig("wiek_auta.png")
    mpl.close()
    print()

    vehicle_damaged = df['Vehicle_Damage']
    print("Srednia uszkodzonych pojazdow " + str(vehicle_damaged.mean()))
    count_vehicle_damaged = vehicle_damaged.value_counts()
    print("Uszkodzone pojazdy 1 - tak, 0 - nie")
    print(count_vehicle_damaged)
    print()

    annual_premiums = df['Annual_Premium']
    print("Srednia kwota ubezpieczenia " + str(annual_premiums.mean()))
    print("Najwyższa kwota ubezpieczenia " + str(annual_premiums.max()))
    print("Najniższa kwota ubezpieczenia " + str(annual_premiums.min()))
    # wykres
    annuals_down_10k = len(list(filter(lambda a: a < 10000, annual_premiums)))
    annuals_10k_to_25k = len(list(filter(lambda a: 10000 <= a < 25000, annual_premiums)))
    annuals_25k_to_40k = len(list(filter(lambda a: 25000 <= a < 40000, annual_premiums)))
    annuals_up_40k = len(list(filter(lambda a: a >= 40000, annual_premiums)))
    x = ["< 10k", "10k - 25k", "25k-40k", "> 40k"]
    y = [annuals_down_10k, annuals_10k_to_25k, annuals_25k_to_40k, annuals_up_40k]
    mpl.bar(x, y)
    mpl.title("Ilość klientów w danym przedziale kwoty ubezpieczenia")
    mpl.xlabel("Kwota")
    mpl.ylabel("Ilosc klientow")
    mpl.savefig("kwota_ubezpieczen.png")
    mpl.close()
    print()

    responses = df['Response']
    print("Srednia wybierajacych ubezpieczenie " + str(responses.mean()))
    count_responses = responses.value_counts()
    print("Wybierajacy ubezpieczenie 1 - tak, 0 - nie")
    print(count_responses)
