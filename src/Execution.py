from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.classification.lazy.knn_adwin import KNNAdwin, KNN
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from src.AdaptiveRandomForest import AdaptiveRandomForest
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging
import timeit
import pandas as pand
import numpy as np
import sys

def run_experiment(dataset = "elec", pre_train_size = 1000):

    # 1. Create a stream
    opt = FileOption("FILE", "OPT_NAME", dataset+".csv", "CSV", False)
    stream = FileStream(opt, -1, 1)
    # 2. Prepare for use
    stream.prepare_for_use()
    # 2. Instantiate the HoeffdingTree classifier

    h = [
            #LeverageBagging(h=KNN(), ensemble_length=2),
            #HoeffdingTree(),
            AdaptiveRandomForest(nb_features=3, nb_trees=20, predict_method="avg", pretrain_size=pre_train_size),
            AdaptiveRandomForest(nb_features=3, nb_trees=40, predict_method="avg", pretrain_size=pre_train_size)
            #AdaptiveRandomForest(m=8, n=25)
         ]
    # 3. Setup the evaluator
    eval1 = EvaluatePrequential(pretrain_size=pre_train_size, output_file='result_'+dataset+'.csv', max_instances=10000, batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=False, plot_options=['performance'])
    # 4. Run
    eval1.eval(stream=stream, classifier=h)


run_experiment()


'''

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def test():
    """Stupid test function"""
    L = []
    for i in range(100):
        L.append(i)

if __name__ == '__main__':
    import timeit
    #print(timeit.timeit("run_experiment()", setup="from __main__ import run_experiment"))


from memory_profiler import memory_usage
mem_usage = memory_usage((run_experiment), include_children=True)
print(mem_usage)

'''

'''times = []
sizes = []
memory_usages = []



results_of_execution = {
    'times': times,
    'sizes': sizes,
    'memory': memory_usages
}

#pprint(results_of_execution)

results_df = pand.DataFrame(results_of_execution)
results_df.index = np.arange(1, len(times)+1)
results_df.to_csv('./data/results/results.csv')'''
