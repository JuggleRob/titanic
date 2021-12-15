import pandas as pd 

# Import dataframe
df = pd.read_csv('titanic-training-dataset.csv')

# Clean Data

# Change Survived datatype from int to bool
df['Survived'] = df['Survived'].astype('bool')

# Remove two rows with null value at Embarked

# Create Title from Name column and remove Name column
df['Title'] = df['Name'].str.split(',', expand = True)[1]
df['Title'] = df['Title'].str.split(' ', expand = True)[1]
df['Title'] = df['Title'].replace(['Lady.', 'Countess.','Capt.', 'Col.','Dr.', 'Major.', 'Rev.', 'Sir.', 'Don.', 'Jonkheer.'], 'Rare.')
df['Title'] = df['Title'].replace('Mlle.', 'Miss.')
df['Title'] = df['Title'].replace('Ms.', 'Miss.')
df['Title'] = df['Title'].replace('Mme.', 'Mrs.')


# Change Pclass to categories
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Title'] = df['Title'].astype('category')

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

def age_groups(x):
    if x < 10:
        return '0-10 years'
    elif x < 20:
        return '10-20 years'
    elif x < 30:
        return '20-30 years'
    elif x < 40:
        return '30-40 years'
    elif x < 50:
        return '40-50 years'
    elif x < 60:
        return '50-60 years'
    else:
        return '>60 years'

def fare_groups(x):
    if x < 10:
        return '0-10 pounds'
    elif x < 20:
        return '10-20 pounds'
    elif x < 30:
        return '20-30 pounds'
    elif x < 40:
        return '30-40 pounds'
    elif x < 50:
        return '40-50 pounds'
    elif x < 100:
        return '50-100 pounds'
    else:
        return '>100 pounds'

df['AgeGroup'] = df['Age'].apply(age_groups)
df['FareGroup'] = df['Fare'].apply(fare_groups)

df.to_csv('titanic-training-dataset-cleaned.csv')