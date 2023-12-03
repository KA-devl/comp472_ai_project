How to run A2 :
**NOTE : you need a file named synonym.csv to run the project.**

In order to run task 1 & 2 of the project, you need to navigate in the task1_2.py file and press the green run button. 
The next step is to uncomment the model that you wish to execute. 
By default, the uncommented model is word2vec-google-news-300. To run the other models, you need to comment the current model and uncomment the model that you wish to execute, and comment the other models.
To generate the plotting graph, you need an analysis.csv file that contains at least 1 model result (in our case, because task 1 is logically before task 2, we expect to have the results of task 1 for google word2vec --> word2vec-google-news-300,3000000,70,79,0.8860759493670886).

After executing all the models, you will obtain the plotting graph that contains the results of all the models that should look like this :

![performance_comparison_graph.png](performance_comparison_graph.png)