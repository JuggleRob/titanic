import pandas as pd

# Impute age based on Pclass
def impute_age(age_pclass):
    age = age_pclass[0]
    pclass = age_pclass[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38 # Average age of people in first class
        elif pclass == 2:
            return 30
        else: 
            return 25
    else:
        return age

def get_age_group(age):
    if age < 10:
        return 0 # 0-10 years 
    elif age < 20:
        return 1 # 10-20 years
    elif age < 30:
        return 2 # 20-30 years 
    elif age < 40:
        return 4 # 30-40 years
    elif age < 50:
        return 5 # 40-50 years
    elif age < 60:
        return 6 # 50-60 years
    else:
        return 7 # >60 years

def get_fare_group(fare):
    if fare < 10:
        return 1 # 0-10 pounds
    elif fare < 20:
        return 2 # 10-20 pounds
    elif fare < 30:
        return 3 # 20-30 pounds
    elif fare < 40:
        return 3 # 30-40 pounds
    elif fare < 50:
        return 4 # 40-50 pounds
    elif fare < 100:
        return 5 # 50-100 pounds
    else:
        return 6 # >100 pounds

def is_alone(family_size):
    if family_size == 1:
        return 1
    else:
        return 0

def get_sex(sex_encoded):
    if sex_encoded == 0:
        return 'female'
    else:
        return 'male'

def encode_sex(sex):
    if sex == 'female':
        return 0
    else:
        return 1