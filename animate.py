import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation

# datatypes = {'initiation_no': np.int32, 'iteration_no': np.int32, 'x': np.float64, 'y':np.float64,
#              'cluster_association': np.int32, 'cluster_center_X': np.float64, 'cluster_center_y':np.float64}

df = pd.read_csv("/home/ravindra/MKSSS_AIT/NLP_batch_Oct_2021/content_notebooks/data/utilities/sample.txt", sep = ",") #, dtype=datatypes)

def get_animation(df, initiation_no):
    """
    :return:
    """
    print("Creating an animation for initiation no.", initiation_no)
    # filter the data for given intiation no
    init_data = df[df['initiation_no'] == initiation_no]
    # print(init_data['x'])

    # create a figure and the objects in it, which will be updated by following "animate" funstion
    fig = plt.figure()
    plt.clf()

    x_margin = (init_data['x'].max() - init_data['x'].min())*0.05
    x_footer = (init_data['x'].max() - init_data['x'].min())*0.1
    xmin = init_data['x'].min() - x_margin - x_footer
    xmax= init_data['x'].max() + x_margin

    y_margin =(init_data['y'].max() - init_data['y'].min()) * 0.05
    y_header = (init_data['y'].max() - init_data['y'].min()) * 0.1
    ymin = init_data['y'].min() - y_margin
    ymax = init_data['y'].max() + y_margin + y_header

    zero_iteration_data = init_data[init_data['iteration_no'] == 0]
    #data_points.set_offsets(np.c_[iteration_data['x'], iteration_data['y']])

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    data_points = ax.scatter(zero_iteration_data['x'], zero_iteration_data['x'], s=5, marker='.')
    cluster_centers = ax.scatter([], [], marker="*", s=20, c='black')

    text_x = xmin + x_margin
    text_y = ymax - y_margin
    footer_x = xmin + x_margin
    footer_y = ymin + y_margin
    title = ax.text(text_x, text_y, "Initiation no. "+str(initiation_no)+ " Iteration no. 0", color='k', fontsize=12)
    footer = ax.text(footer_x, footer_y, "Delta_x. 0"+" "*20+" Delta_y 0", color='k', fontsize=12)


    iterations = list(set(init_data['iteration_no']))
    colors = np.array(['black'] + (['red', 'green', 'blue', 'yellow', 'cyan',  'orange'] * 1000))

    # def init():
    #
    #     init_data = df[df['initiation_no'] == initiation_no]
    #     iteration_data = init_data[init_data['iteration_no'] == 0]
    #     data_points = ax.scatter(iteration_data['x'], iteration_data['y'], s=5, marker='.', c='black')
    #     return data_points

    def animate(i):
        """
        :param
        :return:
        """
        # for iteration in iterations:
        #print(init_data.info())
        #print('-'*50)
        #print("The iteration no is ", i)
        iteration_data = init_data[init_data['iteration_no'] == i]
        #print(iteration_data.columns)
        #print(iteration_data.info())
        data_points.set_offsets(np.c_[iteration_data['x'], iteration_data['y']])
        color = list(colors[np.array(iteration_data['cluster_association'])])
        data_points.set_color(color)
        cluster_centers.set_offsets(np.c_[iteration_data['cluster_center_X'], iteration_data['cluster_center_y']])
        title.set_text("Initiation no. "+str(initiation_no)+ " "*20+" Iteration no. "+str(i))
        delta_x = iteration_data['delta_x'].unique()[0]
        delta_y = iteration_data['delta_y'].unique()[0]
        footer.set_text("Delta_x. "+str(delta_x)+" "*20+" Delta_y "+str(delta_y))
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(iterations), interval=2000)
    return ani

