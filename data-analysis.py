import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataframe
df = pd.read_csv('titanic-training-dataset-cleaned.csv')

# Set grid style
sns.set(style="darkgrid")

#### Creates four plots in one figure  splitted in women/men and absolute numbers/percentages ####
# The top two plots are in absolute numbers of passengers
# The bottom two plots are the percentage of survivors
# The left two plots are women, the right two plots are men
# df = dataframe, feature = sets column that will be used on the x-axis, ylim = sets height of the y-axis for the top two plots
def CreateFigureWomenMen(df, feature, ylim):
    fig, axs = plt.subplots(2,2, sharex=True, num=feature)
    plt.subplots_adjust(wspace=0.016, hspace=0.075)
    
    # The two plots with absolute number of passengers
    survived_women_feature = df[df['Sex'] == 'female'].groupby(feature)["Survived"].count()
    survived_men_feature = df[df['Sex'] == 'male'].groupby(feature)["Survived"].count()
    axs[0,0] = survived_women_feature.plot.bar(ax=axs[0,0], sharey=True, color="#ce546d", title="Women")
    axs[0,1]  = survived_men_feature.plot.bar(ax=axs[0,1], sharey=True, title="Men")
    axs[0,0].set_ylabel("Number of passengers")
    axs[0,0].set_ylim(0, ylim)
    axs[0,1].set_ylim(0, ylim)

    # The two plots with the survival rate
    survival_rate_women_feature = df[df["Sex"] == "female"].groupby(feature)["Survived"].mean()
    survival_rate_men_feature = df[df["Sex"] == "male"].groupby(feature)["Survived"].mean()
    axs[1,0] = survival_rate_women_feature.plot.bar(ax=axs[1,0], color="#ce546d", sharey=True)
    axs[1,1] = survival_rate_men_feature.plot.bar(ax=axs[1,1], sharey=True)
    axs[1,0].set_ylabel("Survival Rate")
    axs[1,0].set_ylim(0.0, 1.0)
    axs[1,1].set_ylim(0.0, 1.0)

#### Creates two plots in one figure ####
# The top plot is the absolute number of passengers per feature
# The bottom plot is the percentage of survivors per feature
# df = dataframe, feature = sets column that will be used on the x-axis, ylim = sets height of the y-axis for the top two plots
def CreateFigure(df, feature, ylim):
    figure_Embarked, axs = plt.subplots(2,1, sharex=True, num=feature)
    plt.subplots_adjust(wspace=0.016, hspace=0.075)

    # figure 4.0 - Number of passengers that survived per embark location
    survived_by_embarked = df.groupby(feature)["Survived"].count()
    axs[0] = survived_by_embarked.plot.bar(ax=axs[0], color=["#339955"])
    axs[0].set_ylabel("Number of passengers")
    axs[0].set_ylim(0, ylim)

    # figure 4.1 - Survival rate for passengers per embark location
    survival_rate_by_embarked = df.groupby(feature)["Survived"].mean()
    axs[1] = survival_rate_by_embarked.plot.bar(ax=axs[1], color=["#339955"])
    axs[1].set_ylabel("Survival Rate")
    axs[1].set_ylim(0.0, 1.0)

# Create the women/men figures
features = ('Pclass', 'AgeGroup', 'FareGroup')
y_limits = (350, 250, 300)
figures = list(map(CreateFigureWomenMen, [df]*3, features, y_limits))

# Create the other figures
features1 = ('Embarked', 'Parch', 'Title')
y_limits1 = (650, 700, 550)
figures.append(list(map(CreateFigure, [df]*3, features1, y_limits1)))

plt.show()