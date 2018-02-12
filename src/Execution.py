from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from src.AdaptiveRandomForest import AdaptiveRandomForest
from skmultiflow.classification.meta.leverage_bagging import LeverageBagging

def run_experiment(dataset="poker", pre_train_size=1000, max_instances=10000, batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=False,
                                plot_options=['performance']):

    # 1. Create a stream
    opt = FileOption("FILE", "OPT_NAME", dataset+".csv", "CSV", False)
    stream = FileStream(opt, -1, 1)
    # 2. Prepare for use
    stream.prepare_for_use()
    # 2. Instantiate the HoeffdingTree classifier

    h = [
            HoeffdingTree(),
            AdaptiveRandomForest(nb_features=6, nb_trees=100, predict_method="avg", pretrain_size=pre_train_size,
                                 delta_d=0.001, delta_w=0.01),
            AdaptiveRandomForest(nb_features=6, nb_trees=5, predict_method="avg", pretrain_size=pre_train_size,
                                 delta_d=0.001, delta_w=0.01)
            #AdaptiveRandomForest(nb_features=3, nb_trees=80, predict_method="avg", pretrain_size=pre_train_size,
              #                   delta_d=0.001, delta_w=0.01)
            #AdaptiveRandomForest(m=8, n=25)
         ]
    # 3. Setup the evaluator
    eval1 = EvaluatePrequential(pretrain_size=pre_train_size, output_file='result_'+dataset+'.csv', max_instances=max_instances,
                                batch_size=batch_size, n_wait=n_wait, max_time=max_time, task_type=task_type, show_plot=show_plot,
                                plot_options=plot_options)
    # 4. Run
    eval1.eval(stream=stream, classifier=h)


run_experiment(dataset="poker", pre_train_size=1000)