import pandas as pd 

# Import dataframe
df = pd.read_csv('titanic-training-dataset.csv')

# Clean Data

# Change Survived datatype from int to bool
df['Survived'] = df['Survived'].astype('bool')

# Remove two rows with null value at Embarked

# Create Title from Name column and remove Name column
df.insert(4, 'Title', '')
df.insert(4, 'TitleFirstname', '')
df['TitleFirstname'] = df['Name'].str.split(',', expand = True)[1]
df['Title'] = df['TitleFirstname'].str.split(' ', expand = True)[1]
df = df.drop(columns = ['TitleFirstname', 'Name'])
df['Title'] = df['Title'].replace(['Lady.', 'Countess.','Capt.', 'Col.','Dr.', 'Major.', 'Rev.', 'Sir.', 'Don.', 'Jonkheer.'], 'Rare.')
df['Title'] = df['Title'].replace('Mlle.', 'Miss.')
df['Title'] = df['Title'].replace('Ms.', 'Miss.')
df['Title'] = df['Title'].replace('Mme.', 'Mrs.')


# Change Pclass to categories
df['Pclass'] = df['Pclass'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Sex'] = df['Sex'].astype('category')
df['Title'] = df['Title'].astype('category')

# Remove ticket column
df.drop(columns = ['Ticket'])

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

df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)
