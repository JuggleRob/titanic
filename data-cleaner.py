import pandas as pd 
from prompt_toolkit import prompt

input = prompt("Enter 1 to clean the training dataset. Enter 2 to clean the test dataset. Choice: ")
if input == "1":
    url = "titanic-training-dataset"
else:
    url = "titanic-test-dataset"

# Import dataframe
df = pd.read_csv(url + ".csv")

# Create Title from Name column and remove Name column
# 1 = Miss , 2 = Mrs , 3 = Mr , 4 = Master , 5 = Other
df['Title'] = df['Name'].str.split(',', expand = True)[1]
df['Title'] = df['Title'].str.split(' ', expand = True)[1]
df['Title'] = df['Title'].replace(['Mlle.', 'Ms.', 'Miss.'], 1)
df['Title'] = df['Title'].replace(['Mrs.', 'Mme.'], 2)
df['Title'] = df['Title'].replace('Mr.', 3)
df['Title'] = df['Title'].replace('Master.', 4)
df['Title'] = df['Title'].replace(['Lady.', 'Countess.','Capt.', 'Col.','Dr.', 'Major.', 'Rev.', 'Sir.', 'Don.', 'Jonkheer.', 'Dona.'], 5)

# Encode Sex to: female = 0 , male = 1
df['Sex'] = df['Sex'].replace('female', 0)
df['Sex'] = df['Sex'].replace('male', 1)

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

# Data analysis
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

def age_groups(age):
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

def fare_groups(fare):
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

# Create two new features
df['AgeGroup'] = df['Age'].apply(age_groups)
df['FareGroup'] = df['Fare'].apply(fare_groups)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = df['FamilySize'].apply(is_alone)

# Delete Cabin, Name, Fare, Ticket and PassengerId column
df.drop(['Cabin', 'Name', 'Ticket', 'Fare', 'PassengerId'], 1, inplace=True)

# Remove two rows with null value at Embarked
df.dropna(inplace=True)

# Encode Embarked: S = 0 , C = 1, Q = 2
df['Embarked'] = df['Embarked'].replace(['S','C','Q'], [0,1,2])

df.to_csv(url + "-cleaned.csv", index=False)