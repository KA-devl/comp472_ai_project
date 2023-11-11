import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    #Question 1 a,b
    # Load Penguin & abalone  dataset
    penguin_data = pd.read_csv('penguins.csv')
    abalone_data = pd.read_csv('abalone.csv')

    # Convert categorical features to 1-hot vectors
    penguin_data = pd.get_dummies(penguin_data, columns=['island', 'sex'])

    #Question 2 Plot the percentage of the instances in each output class and store the graphic in a file called
    # For Penguin dataset
    penguin_class_distribution = penguin_data['species'].value_counts(normalize=True)
    penguin_class_distribution.plot(kind='bar')
    plt.savefig('penguin-classes.jpg')
    ##convert jpg to gif
    image = imageio.imread('penguin-classes.jpg')
    imageio.mimsave('penguin-classes.gif', [image], fps=1)


    # For Abalone dataset
    abalone_class_distribution = abalone_data['Type'].value_counts(normalize=True)
    abalone_class_distribution.plot(kind='bar')
    plt.savefig('abalone-classes.jpg')
    image = imageio.imread('abalone-classes.jpg')
    imageio.mimsave('abalone-classes.gif', [image], fps=1)


    #Question 3
    # Split the datasets into training and test sets
    # For Penguin dataset
    #features are all columns except species
    X_penguin = penguin_data.drop(columns=['species'])
    #species is the target
    y_penguin = penguin_data['species']
    X_train_penguin, X_test_penguin, y_train_penguin, y_test_penguin = train_test_split(X_penguin, y_penguin)

    # For Abalone dataset
    #features are all columns except Type
    X_abalone = abalone_data.drop(columns=['Type'])
    #Type is the target
    y_abalone = abalone_data['Type']
    X_train_abalone, X_test_abalone, y_train_abalone, y_test_abalone = train_test_split(X_abalone, y_abalone)

    #Question 4
    # (a) Base-DT
    #Penguins
    ##create DT classifier
    base_dt_penguins = DecisionTreeClassifier()
    ##fit the model
    base_dt_penguins.fit(X_train_penguin, y_train_penguin)
    ##plot tree
    tree.plot_tree(base_dt_penguins)
    plt.savefig('base-dt-penguins.jpg')

    #Abalone
    ##create DT classifier
    base_dt_abalone = DecisionTreeClassifier()
    ##fit the model
    base_dt_abalone.fit(X_train_abalone, y_train_abalone)
    ##plot tree
    #Change fontsize if needed
    tree.plot_tree(base_dt_abalone, max_depth=2, fontsize=3)
    plt.savefig('base-dt-abalone.jpg')


    # # (b) Top-DT
    ##Pinguin
    params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 10],
        'min_samples_split': [2, 6, 8]
    }
    top_dt_penguins = GridSearchCV(DecisionTreeClassifier(), params)
    ##fit the model
    top_dt_penguins.fit(X_train_penguin, y_train_penguin)
    best_top_dt = top_dt_penguins.best_estimator_

    ##plot tree
    plt.figure()
    tree.plot_tree(best_top_dt, fontsize=4)
    plt.savefig('top-dt-penguins.jpg')

    ##Abalone
    top_dt_abalone = GridSearchCV(DecisionTreeClassifier(), params)
    top_dt_abalone.fit(X_train_abalone, y_train_abalone)
    best_top_dt2 = top_dt_abalone.best_estimator_
    plt.figure()
    tree.plot_tree(best_top_dt2, fontsize=4)
    plt.savefig('top-dt-abalone.jpg')


    ## c) base mlp
    ## d) top mlp
    ##Question 5) performance





