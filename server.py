from multiprocessing import Process
import json
import os

from flask import Flask
from flask import request, make_response

import pandas as pd

import Visualizer


app = Flask(__name__)
vis = Visualizer.Visualizer()


@app.route('/<arg>', methods=['POST'])
def hello(arg):
    visualizing_data = json.loads(arg)
    if visualizing_data['plot_type'] == 'euclidean_scatter':
        vis.make_euclidean_scatter_plot(visualizing_data['estimator_name'],
                                        visualizing_data['data'],
                                        visualizing_data['picture_name'],
                                        visualizing_data['predicted_points'],
                                        visualizing_data['center_points']
                                        )
    elif visualizing_data['plot_type'] == 'compare_plot':
        vis.make_compare_plot(visualizing_data['x_points'],
                              visualizing_data['y_points'],
                              visualizing_data['picture_name']
                              )
    elif visualizing_data['plot_type'] == 'simple_graph':
        vis.make_simple_graph(visualizing_data['points_x'],
                              visualizing_data['number_of_points']
                              )
    elif visualizing_data['plot_type'] == 'heatmap':
        dataframe_matrix = pd.read_json(visualizing_data['dataframe_matrix'])
        print('go_to_make_image')
        make_process(vis.make_heatmap,dataframe_matrix,
                         visualizing_data['ages_columns'],
                         visualizing_data['picture_name'],
                         visualizing_data['cluster_name'])
        print('image ready')
    elif visualizing_data['plot_type'] == 'pairplot':
        vis.make_pairplot(visualizing_data['dataframe_matrix'],
                          visualizing_data['ages_columns'],
                          visualizing_data['picture_name']
                          )
    elif visualizing_data['plot_type'] == 'overfit_check':
        vis.make_overfit_check_plot(visualizing_data['train_score_list'],
                                    visualizing_data['test_score_list'],
                                    visualizing_data['values_range'],
                                    visualizing_data['cluster_name'],
                                    visualizing_data['compare_flag']
                                    )
    elif visualizing_data['plot_type'] == 'confusion_matrix':
        vis.make_confusion_matrix(visualizing_data['input_labels'],
                                  visualizing_data['predicted_labels'],
                                  visualizing_data['train_length']
                                  )
    elif visualizing_data['plot_type'] == 'distr_hist':
        vis.distribution_hist(visualizing_data['predictor_values'],
                              visualizing_data['cluster_name'],
                              visualizing_data['predictor_name'],
                              visualizing_data['predictor_probe']
                              )
    else:
        pass
    return make_response()

def make_process(input_method, *args):
    p = Process(target=input_method, args=(*args,))
    p.start()


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


