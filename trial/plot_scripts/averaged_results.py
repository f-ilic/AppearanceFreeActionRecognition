import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from os.path import join
import random
# from trial.plot_scripts.secondary_results import per_user_results

from trial.plot_scripts.util import collect_all_user_data, get_usefull_data_from_df

def subplot_overview_correct_false(data, ax=None):
    if ax is None:
        ax = plt.gca()
    
    num_correct = sum(data.Correct)
    num_total = len(data.Correct)
    num_wrong = num_total - num_correct
    labels = 'Correct', 'Wrong'
    plotdata = [num_correct, num_wrong]
    subplt = ax.pie(plotdata, explode=(0.1, 0.1), shadow=True, startangle=80, labels=labels, autopct='%1.1f%%',textprops={'fontsize': 14, 'fontweight':'bold'}, colors=('C0', 'red'), wedgeprops={"edgecolor":"k",'linewidth': 2, 'antialiased': True})
    ax.set_title("Top 1")

    for autotext in subplt[2]:
        autotext.set_color('white')
    return subplt


def subplot_confusion_matrix(data, ax=None):
    class ConfusionMatrix(object):
        def __init__(self, n_classes, labels=None):
            self.n_classes = n_classes
            self.mat = torch.zeros(n_classes, n_classes, requires_grad=False)
            self.labels = labels

        def update(self, preds, labels):
            for p, t in zip(preds, labels):
                self.mat[t, p] += 1

    predictions, gtlabels, labels, n_classes = get_usefull_data_from_df(data)
    label_to_idx = {l: idx for idx, l in enumerate(labels)}
    confusionmatrix = ConfusionMatrix(n_classes, labels)

    if ax is None:
        ax = plt.gca()

    for p,g in zip(data.Prediction, data.GroundTruth):
        confusionmatrix.update([label_to_idx[p]], [label_to_idx[g]])
    
    fontsize=15
    label_angle=45
    mat = confusionmatrix.mat

    total_samples = mat.sum(axis=1)
    mat = mat / total_samples[:, None]
    mat = torch.nan_to_num(mat, nan=0)

    subplt = ax.matshow(mat, cmap=plt.cm.Blues)

    threshold = mat.max() / 2.
    for (i, j), z in np.ndenumerate(mat):
        color = "white" if z > threshold else "black"
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', color=color, fontsize=fontsize)

    ax.set_xticks(range(0, n_classes))
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(labels, rotation=label_angle, fontsize=fontsize)
    ax.set_yticks(range(0, n_classes))
    ax.set_yticklabels(labels, rotation=label_angle, fontsize=fontsize)

    ax.set_ylabel('True Class')
    ax.yaxis.set_label_position("right") 
    ax.set_xlabel('Predicted Class')
    ax.set_title("Confusion Matrix")
    return subplt


def subplot_confusion_histogram(data, ax=None):
    predictions, gtlabels, labels, n_classes = get_usefull_data_from_df(data)

    if ax is None:
        ax = plt.gca()


    num_correct_per_class = []
    num_wrong_per_class = []

    for lbl in labels:
        d = data.loc[data.GroundTruth==lbl]
        num_correct_per_class.append(len(d.loc[d.GroundTruth==d.Prediction]))
        num_wrong_per_class.append(len(d.loc[d.GroundTruth!=d.Prediction]))

    width = 0.5       # the width of the bars: can also be len(x) sequence


    subplt = ax.bar(labels, num_correct_per_class, width, label='Correct', edgecolor='black')
    ax.bar(labels, num_wrong_per_class, width, bottom=num_correct_per_class, label='Wrong', color='red', edgecolor='black')
    ax.set_xticklabels(labels, rotation=45, fontsize=15)
    bottom, top = ax.get_ylim()  # return the current ylim
    ax.set_ylim([bottom, top+(top*0.1)])
    ax.set_ylabel('# Videos')
    ax.set_title('Class confusion Histogram')
    ax.legend()

    return subplt 



def subplot_decision_duration_chronological(data, ax=None):
    duration = list(data.Duration)
    correct = list(data.Correct)

    colors = []
    for i in correct:
        c = 'C0' if i==1 else 'red'
        colors.append(c)

    if ax is None:
        ax = plt.gca()

    subplt = ax.scatter(range(len(duration)), duration, c=colors)
    ax.set_ylabel('Viewing Duration (seconds)')
    ax.set_xlabel('Video-number shown')
    ax.set_title('How long to decide')
    ax.legend()

    return subplt 

def subplot_decision_duration_per_class(data, ax=None, ylimtop=None):
    predictions, gtlabels, labels, n_classes = get_usefull_data_from_df(data)

    if ax is None:
        ax = plt.gca()

    duration_per_class = {}
    for i, lbl in enumerate(labels):
        durations = list(data.loc[data.GroundTruth==lbl].Duration)
        duration_per_class[lbl] = durations

        tmp = list(data.loc[data.GroundTruth==lbl].Correct)
        colors = []
        for tt in tmp:
            c = 'C0' if tt==1 else 'red'
            colors.append(c)
        
        x = np.random.normal(i+1, 0.08, size=len(durations))
        ax.scatter(x, durations, alpha=0.15,  s=20, c=colors)
        # ax.plot(x, durations, 'bo')

    ax.boxplot(duration_per_class.values(), widths=0.3, vert=True, showfliers=False, boxprops=dict(linewidth=2, color='black'), whiskerprops=dict(linestyle='-',linewidth=2.0, color='black')) 
    ax.set_xticklabels(duration_per_class.keys(), fontsize=15)
    

    red_patch = mpatches.Patch(color='red', label='Wrong')
    blue_patch = mpatches.Patch(color='C0', label='Correct')

    ax.legend(handles=[blue_patch, red_patch], loc='upper right')
    ax.set_ylim([0, ylimtop])
    ax.set_ylabel('Viewing Duration (seconds)')
    # ax.set_xlabel('Video-number shown')
    ax.set_title('Decision speed')
    ax.set_xticklabels(labels, rotation=45)

    # ax.legend()

    return 


def main_from_csv(filename):
    df_all = pd.read_csv(filename)
    main(df_all)

def main(dataframe, save_dir=None):

    df_all = dataframe.copy()
    df_all = df_all.loc[df_all.Trainmode==0]

    # TODO: SET THIS MAX YLIM TO SOMETHING REASONABLE
    max_decision_time = round(max(list(df_all.Duration))) + 2
    max_decision_time = 30
    modalities = set(df_all.Modality)
    
    modalities_nicename = {
        "ucf5": "UCF5",
        "afd5": "AFD5",
        # "ucf5rgbflow": "DONT USE RGBFLOW"
    }

    for modality in ['ucf5', 'afd5']:
        df = df_all.loc[df_all.Modality==modality]
        f, axes = plt.subplots(1, 4, figsize=(16, 5))
        (ax1, ax2, ax3, ax4) = axes.ravel()
        subplot_overview_correct_false(df, ax=ax1)
        subplot_confusion_matrix(df, ax=ax2)
        subplot_decision_duration_per_class(df, ax=ax3, ylimtop=max_decision_time)
        subplot_confusion_histogram(df, ax=ax4)

        f.suptitle(f"{modalities_nicename[set(df.Modality).pop()]}")
        plt.show(block=False)

        if save_dir != None:
            f.savefig(f'{join(save_dir, modality + ".pdf")}')
        f.tight_layout()
    plt.tight_layout()
    plt.show(block=True)




if __name__ == "__main__":
    root_dir = '/home/f/projects/MotionPaper/trial/output/'
    root_output_dir = '/home/f/projects/MotionPaper/trial/output'
    df = collect_all_user_data(root_dir)

    main(df, save_dir=root_dir)
    # per_user_results()
