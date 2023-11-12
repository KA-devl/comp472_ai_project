import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    #Question 1 a,b =========================================================================================
    # Load Penguin & abalone  dataset
    penguin_data = pd.read_csv('penguins.csv')
    abalone_data = pd.read_csv('abalone.csv')

    # Convert categorical features to 1-hot vectors
    penguin_data = pd.get_dummies(penguin_data, columns=['island', 'sex'])

    #Question 2 =========================================================================================
    # Plot the percentage of the instances in each output class and store the graphic in a file called
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


    #Question 3 =========================================================================================
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

    #Question 4 =========================================================================================
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


    # # (b) Top-DT =========================================================================================
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


    ## c) BASE MLP=========================================================================================
    ##Penguin
    ##create MLP classifier with a Multi-Layered Perceptron with 2 hidden layers of 100+100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values for the rest of the parameters
    base_mlp_penguins = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    base_mlp_penguins.fit(X_train_penguin, y_train_penguin)

    ##Abalone
    ##disabled because causes ConverganceWarning
    # base_mlp_abalone = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd')
    # base_mlp_abalone.fit(X_train_abalone, y_train_abalone)


    ## d) TOP MLP=========================================================================================
    ##Hyper parameters activation function: sigmoid, tanh and relu,
    # 2 network architectures of your choice: for eg 2 hidden layers with 30 + 50 nodes, 3 hidden layers with 10 + 10 + 10 solver: adam and stochastic gradient descent
    #Penguin
    params = {
        'activation': ['logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
        'solver': ['adam', 'sgd']
    }
    top_mlp_penguins = GridSearchCV(MLPClassifier(), params)
    top_mlp_penguins.fit(X_train_penguin, y_train_penguin)

    ##Abalone
    ##disabled because causes ConverganceWarning
    # top_mlp_abalone = GridSearchCV(MLPClassifier(), params)
    # top_mlp_abalone.fit(X_train_abalone, y_train_abalone)

    ##Question 5) performance measure ========================================================
    ##For each of the 4 classifiers above 4(a), 4(b), 4(c) and 4(d), append the following information in a file called
    ##Penguin
    #A) clear string describing the model, hyper parameters values that were changed
    penguin_y_pred_base_dt = base_dt_penguins.predict(X_test_penguin)
    print("================================= Base DT (penguins) ============================================:")
    print("Hyper parameters values that were changed :")
    print(base_dt_penguins.get_params())
    #B) the confusion matrix
    penguin_confusion_matrix_base_dt = pd.crosstab(y_test_penguin, penguin_y_pred_base_dt, rownames=['Actual'], colnames=['Predicted'])
    print("Confusion matrix for base DT :")
    print(penguin_confusion_matrix_base_dt)
    #C) The precision, recall, and F1-score for each class
    print("Precision, recall, and F1-score for each class :")
    print(classification_report(y_test_penguin, penguin_y_pred_base_dt))
    #D) The accuracy, macro-average F1-score, and weighted-average F1-score
    print('Accuracy: ', base_dt_penguins.score(X_test_penguin, y_test_penguin))
    print('Macro-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_base_dt, output_dict=True)['macro avg']['f1-score'])
    print('Weighted-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_base_dt, output_dict=True)['weighted avg']['f1-score'])

    print("================================= Top DT (penguins) ============================================:")
    penguin_y_pred_top_dt = best_top_dt.predict(X_test_penguin)
    print("Best parameters found by gridsearch :")
    print(top_dt_penguins.best_params_)
    penguin_confusion_matrix_top_dt = pd.crosstab(y_test_penguin, penguin_y_pred_top_dt, rownames=['Actual'], colnames=['Predicted'])
    print("Confusion matrix for top DT :")
    print(penguin_confusion_matrix_top_dt)
    print("Precision, recall, and F1-score for each class :")
    print(classification_report(y_test_penguin, penguin_y_pred_top_dt))
    print('Accuracy: ', best_top_dt.score(X_test_penguin, y_test_penguin))
    print('Macro-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_top_dt, output_dict=True)['macro avg']['f1-score'])
    print('Weighted-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_top_dt, output_dict=True)['weighted avg']['f1-score'])

    print("================================= Base MLP (penguins) ============================================:")
    penguin_y_pred_base_mlp = base_mlp_penguins.predict(X_test_penguin)
    print("Hyper parameters values that were changed :")
    print(base_mlp_penguins.get_params())
    print("Confusion matrix for base MLP :")
    print(confusion_matrix(y_test_penguin, penguin_y_pred_base_mlp))
    print("Precision, recall, and F1-score for each class :")
    print(classification_report(y_test_penguin, penguin_y_pred_base_mlp, zero_division=0))
    print('Accuracy: ', base_mlp_penguins.score(X_test_penguin, y_test_penguin))
    # zero_devision issue
    # print('Macro-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_base_mlp, output_dict=True)['macro avg']['f1-score'])
    # print('Weighted-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_base_mlp, output_dict=True)['weighted avg']['f1-score'])

    print("================================= Top MLP (penguins) ============================================:")
    penguin_y_pred_top_mlp = top_mlp_penguins.predict(X_test_penguin)
    print("Best parameters found by gridsearch :")
    print(top_mlp_penguins.best_params_)
    print("Confusion matrix for top MLP :")
    print(confusion_matrix(y_test_penguin, penguin_y_pred_top_mlp))
    print("Precision, recall, and F1-score for each class :")
    print(classification_report(y_test_penguin, penguin_y_pred_top_mlp, zero_division=0))
    print('Accuracy: ', top_mlp_penguins.score(X_test_penguin, y_test_penguin))
    # zero_devision issue
    # print('Macro-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_top_mlp, output_dict=True)['macro avg']['f1-score'])
    # print('Weighted-average F1-score: ', classification_report(y_test_penguin, penguin_y_pred_top_mlp, output_dict=True)['weighted avg']['f1-score'])

    ##TO DO
    ##1. Abalone
    ##2. Write to file
    ##3. Do 5 iterations and take average (Question 6)