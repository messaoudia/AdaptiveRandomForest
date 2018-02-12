# Adaptive Random Forest
Article used : https://www.researchgate.net/publication/317579226_Adaptive_random_forests_for_evolving_data_stream_classification
By Albert Bifet & al.

Project done by
- Jeremy AECK
- Amin MESSAOUDI
- Nedeljko RADULOVIC


--- What is in the project ?

In the project you will find:

|
|----- Presentation : Adaptive_Random_Forest_JeremyAECK_AminMESSAOUDI_NedeljkoRADULOVIC
|
|----- src :
        |----- Adaptive Random Forest implementation : AdaptiveRandomForest.py
        |----- Adaptive Hoeffding Tree implementation : ARFHoeffdingTree.py
        |----- A file for running tests : Execution.py
        |----- A file for showing plots : run_plot.py
        |----- Some datasets to try : poker.csv, covtype.csv and elec.csv


--- How to run it ?
Go into execution file. In order to run an experiment, please use the function
def run_experiment(dataset="poker", pre_train_size=1000, max_instances=10000, batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=False,
                                plot_options=['performance']):

To try another dataset pass another parameter to the function