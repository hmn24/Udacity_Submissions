###########################################
### Taken from Udacity customer_segments project's visual.py with modifications made to evaluate function
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches

def plot_scatter_matrix(df):
    """
	To plot the scatter_matrix for analytical purposes, where the diagonal would be populated by kernel density estimation
	"""
    
    axs = pd.scatter_matrix(df, alpha = 0.3, figsize = (15,10), diagonal = 'kde')

    n = len(df.columns)

    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50
    
    return None

def print_cm_single(cm, labels, title_name):
    """
	Print a single confusion matrix
	With credits from https://www.tarekatwan.com/index.php/2017/12/how-to-plot-a-confusion-matrix-in-python/
    """

    pl.clf()
    pl.imshow(cm, interpolation='nearest', cmap=pl.cm.Wistia)
    classNames = labels
    pl.title(title_name)
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    pl.xticks(tick_marks, classNames)
    pl.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            pl.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), horizontalalignment='center', verticalalignment='center')
    pl.show()
    
def print_cm_mult(cm, sample_labels, labels, title_name):
    """
    Print multiple confusion matrices
    """

    _, ax = pl.subplots(1, len(sample_labels), figsize = (11,7))    

    for m, sample_title in enumerate(sample_labels):
        ax[m].imshow(cm[m], interpolation='nearest', cmap=pl.cm.Wistia)
        classNames = labels
        ax[m].set_title(sample_title)
        ax[m].set_ylabel('True label')
        ax[m].set_xlabel('Predicted label')      
        tick_marks = np.arange(len(classNames))
        ax[m].set_xticks(tick_marks)
        ax[m].set_yticks(tick_marks)
        ax[m].set_xticklabels(classNames)
        ax[m].set_yticklabels(classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                ax[m].text(j,i, str(s[i][j])+" = "+str(cm[m][i][j]), horizontalalignment='center', verticalalignment='center')
    pl.tight_layout()
    pl.show()
	
def evaluate(results):
    """
    Visualization code to display results of various learners.
    """

	# Taken from Udacity finding_donor project's visual.py
    # Create figure
    fig, ax = pl.subplots(2, 3, figsize = (11,7))

    # Constants
    bar_width = 0.3
    colors = ['#A00000','#00A0A0','#00A000']
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()):
        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):
            for i in np.arange(3):
                
                # Creative plot code
                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.45, 1.45, 2.45])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    ax[0, 0].set_ylabel("Time (in seconds)")
    ax[0, 1].set_ylabel("Accuracy Score")
    ax[0, 2].set_ylabel("F-score")
    ax[1, 0].set_ylabel("Time (in seconds)")
    ax[1, 1].set_ylabel("Accuracy Score")
    ax[1, 2].set_ylabel("F-score")
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[0, 1].set_title("Accuracy Score on Training Subset")
    ax[0, 2].set_title("F-score on Training Subset")
    ax[1, 0].set_title("Model Predicting")
    ax[1, 1].set_title("Accuracy Score on Testing Set")
    ax[1, 2].set_title("F-score on Testing Set")
    
    # Set y-limits for score panels
    ax[0, 1].set_ylim((0, 1))
    ax[0, 2].set_ylim((0, 1))
    ax[1, 1].set_ylim((0, 1))
    ax[1, 2].set_ylim((0, 1))

    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \
               loc = 'upper center', borderaxespad = 0., ncol = 3, fontsize = 'x-large')
    
    # Aesthetics
    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, y = 1.10)
    pl.tight_layout()
    pl.show()
	
def feature_plot(importances, X_train, y_train):
    """
	Visualisation code to plot feature importances plot
	"""

    # Modified from Udacity finding_donor project's visual.py
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:len(indices)]]
    values = importances[indices][:len(indices)]

    # Create the plot
    fig = pl.figure(figsize = (20,len(indices)))
    pl.title("Normalized Weights for Features Importance Plot", fontsize = 16)
    pl.bar(np.arange(len(indices)), values, width = 0.6, align="center", color = '#00A000', \
        label = "Feature Weight")
    pl.bar(np.arange(len(indices)) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
        label = "Cumulative Feature Weight")
    pl.xticks(np.arange(len(indices)), columns, fontsize=16)
    pl.yticks(fontsize=15)
    pl.xlim((-0.5, len(indices)-0.5))
    pl.ylabel("Weight", fontsize = 20)
    pl.xlabel("Feature", fontsize = 20)

    pl.legend(loc = 'upper left', fontsize=20)
    pl.tight_layout()
    pl.show()  