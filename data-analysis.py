import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataframe
df = pd.read_csv('titanic-training-dataset-cleaned.csv')

sns.set(style="darkgrid")

# Contains the figures 1.0 and 1.1 
figure_Pclass, axs = plt.subplots(2,2, sharex=True, num="Pclass")

# figure 1.0 - Number of women/men that survived per class
survived_women_by_class = df[df["Sex"] == "female"].groupby("Pclass")["Survived"].count()
survived_men_by_class = df[df["Sex"] == "male"].groupby("Pclass")["Survived"].count()
axs[0,0] = survived_women_by_class.plot.bar(ax=axs[0,0], sharey=True, color="#ce546d", title="Women")
axs[0,1]  = survived_men_by_class.plot.bar(ax=axs[0,1], sharey=True, title="Men")
axs[0,0].set_ylabel("Number of passengers")
axs[0,0].set_ylim(0, 350)
axs[0,1].set_ylim(0, 350)

# figure 1.1 - Survival rate for women/men per class
survival_rate_women_by_class = df[df["Sex"] == "female"].groupby("Pclass")["Survived"].mean()
survival_rate_men_by_class = df[df["Sex"] == "male"].groupby("Pclass")["Survived"].mean()
axs[1,0] = survival_rate_women_by_class.plot.bar(ax=axs[1,0], color="#ce546d", sharey=True)
axs[1,1] = survival_rate_men_by_class.plot.bar(ax=axs[1,1], sharey=True)
axs[1,0].set_ylabel("Survival Rate")
axs[1,0].set_ylim(0.0, 1.0)
axs[1,1].set_ylim(0.0, 1.0)

# Contains the figures 2.0 and 2.1
figure_AgeGroup, axs = plt.subplots(2,2, sharex=True, num="Age Group")

# figure 2.0 - Number of women/men that survived per age group
survived_women_by_AgeGroup = df[df["Sex"] == "female"].groupby("AgeGroup")["Survived"].count()
survived_men_by_AgeGroup = df[df["Sex"] == "male"].groupby("AgeGroup")["Survived"].count()
axs[0,0] = survived_women_by_AgeGroup.plot.bar(ax=axs[0,0], sharey=True, color="#ce546d", title="Women")
axs[0,1]  = survived_men_by_AgeGroup.plot.bar(ax=axs[0,1], sharey=True, title="Men")
axs[0,0].set_ylabel("Number of passengers")
axs[0,0].set_ylim(0, 250)
axs[0,1].set_ylim(0, 250)

# figure 2.1 - Survival rate for women/men per age group
survival_rate_women_by_AgeGroup = df[df["Sex"] == "female"].groupby("AgeGroup")["Survived"].mean()
survival_rate_men_by_AgeGroup = df[df["Sex"] == "male"].groupby("AgeGroup")["Survived"].mean()
axs[1,0] = survival_rate_women_by_AgeGroup.plot.bar(ax=axs[1,0], color="#ce546d", sharey=True)
axs[1,1] = survival_rate_men_by_AgeGroup.plot.bar(ax=axs[1,1], sharey=True)
axs[1,0].set_ylabel("Survival Rate")
axs[1,0].set_ylim(0.0, 1.0)
axs[1,1].set_ylim(0.0, 1.0)

# Contains the figures 3.0 and 3.1
figure_FareGroup, axs = plt.subplots(2,2, sharex=True, num="Fare Group")

# figure 3.0 - Number of women/men that survived per fare group
survived_women_by_FareGroup = df[df["Sex"] == "female"].groupby("FareGroup")["Survived"].count()
survived_men_by_FareGroup = df[df["Sex"] == "male"].groupby("FareGroup")["Survived"].count()
axs[0,0] = survived_women_by_FareGroup.plot.bar(ax=axs[0,0], sharey=True, color="#ce546d", title="Women")
axs[0,1]  = survived_men_by_FareGroup.plot.bar(ax=axs[0,1], sharey=True, title="Men")
axs[0,0].set_ylabel("Number of passengers")
axs[0,0].set_ylim(0, 300)
axs[0,1].set_ylim(0, 300)

# figure 3.1 - Survival rate for women/men per fare group
survival_rate_women_by_FareGroup = df[df["Sex"] == "female"].groupby("FareGroup")["Survived"].mean()
survival_rate_men_by_FareGroup = df[df["Sex"] == "male"].groupby("FareGroup")["Survived"].mean()
axs[1,0] = survival_rate_women_by_FareGroup.plot.bar(ax=axs[1,0], color="#ce546d", sharey=True)
axs[1,1] = survival_rate_men_by_FareGroup.plot.bar(ax=axs[1,1], sharey=True)
axs[1,0].set_ylabel("Survival Rate")
axs[1,0].set_ylim(0.0, 1.0)
axs[1,1].set_ylim(0.0, 1.0)

# Contains the figures 4.0 and 4.1
figure_Embarked, axs = plt.subplots(2,1, sharex=True, num="Embarked")

# figure 4.0 - Number of passengers that survived per embark location
survived_by_embarked = df.groupby("Embarked")["Survived"].count()
axs[0] = survived_by_embarked.plot.bar(ax=axs[0], color=["#338899", "#811243", "#DD8211"])
axs[0].set_ylabel("Number of passengers")
axs[0].set_ylim(0, 650)

# figure 4.1 - Survival rate for passengers per embark location
survival_rate_by_embarked = df.groupby("Embarked")["Survived"].mean()
axs[1] = survival_rate_by_embarked.plot.bar(ax=axs[1], color=["#338899", "#811243", "#DD8211"])
axs[1].set_ylabel("Survival Rate")
axs[1].set_ylim(0.0, 1.0)

# Contains the figures 5.0 and 5.1
figure_sibsp, axs = plt.subplots(2,1, sharex=True, num="Siblings and Spouses")

# figure 5.0 - Number of passengers that survived per number of siblings and spouses on board
survived_by_sibsp = df.groupby("SibSp")["Survived"].count()
axs[0] = survived_by_sibsp.plot.bar(ax=axs[0], color=["#339955"])
axs[0].set_ylabel("Number of passengers")
axs[0].set_ylim(0, 700)

# figure 5.1 - Survival rate for passengers per number of siblings and spouses on board
survival_rate_by_sibsp = df.groupby("SibSp")["Survived"].mean()
axs[1] = survival_rate_by_sibsp.plot.bar(ax=axs[1], color=["#339955"])
axs[1].set_ylabel("Survival Rate")
axs[1].set_ylim(0.0, 1.0)

# Contains the figures 6.0 and 6.1
figure_parch, axs = plt.subplots(2,1, sharex=True, num="Parents and Children")

# figure 6.0 - Number of passengers that survived per number of parents/children on board
survived_by_parch = df.groupby("Parch")["Survived"].count()
axs[0] = survived_by_parch.plot.bar(ax=axs[0], color=["#339955"])
axs[0].set_ylabel("Number of passengers")
axs[0].set_ylim(0, 700)

# figure 6.1 - Survival rate for passengers per number of parents/children on board
survival_rate_by_parch = df.groupby("Parch")["Survived"].mean()
axs[1] = survival_rate_by_parch.plot.bar(ax=axs[1], color=["#339955"])
axs[1].set_ylabel("Survival Rate")
axs[1].set_ylim(0.0, 1.0)

# Contains the figures 7.0 and 7.1
figure_title, axs = plt.subplots(2,1, sharex=True, num="Title")

# figure 7.0 - Number of passengers that survived per title
survived_by_title = df.groupby("Title")["Survived"].count()
axs[0] = survived_by_title.plot.bar(ax=axs[0], color=["#339955"])
axs[0].set_ylabel("Number of passengers")
axs[0].set_ylim(0, 550)

# figure 7.1 - Survival rate for passengers per title
survival_rate_by_title = df.groupby("Title")["Survived"].mean()
axs[1] = survival_rate_by_title.plot.bar(ax=axs[1], color=["#339955"])
axs[1].set_ylabel("Survival Rate")
axs[1].set_ylim(0.0, 1.0)

plt.show()