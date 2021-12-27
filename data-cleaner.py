import pandas as pd 
from titanic_methods import get_age_group, get_fare_group, is_alone, impute_age, encode_sex
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

# Data analysis
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

# Encode Sex to: female = 0 , male = 1
df['Sex'] = df['Sex'].apply(encode_sex)

# Create four new features
df['AgeGroup'] = df['Age'].apply(get_age_group)
df['FareGroup'] = df['Fare'].apply(get_fare_group)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = df['FamilySize'].apply(is_alone)

# Delete Cabin, Name, Fare, Ticket and PassengerId column
df.drop(['Cabin', 'Name', 'Ticket', 'Fare', 'PassengerId'], 1, inplace=True)

# Remove two rows with null value at Embarked
df.dropna(inplace=True)

# Encode Embarked: S = 0 , C = 1, Q = 2
df['Embarked'] = df['Embarked'].replace(['S','C','Q'], [0,1,2])

df.to_csv(url + "-cleaned.csv", index=False)