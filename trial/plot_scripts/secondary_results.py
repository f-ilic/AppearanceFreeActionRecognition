import os
from re import I
from click import style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import seaborn as sns
from util import collect_all_user_data, get_user_csv
# from trial.plot_scripts import collect_all_user_data, get_user_csv


def per_user_results():
    root_dir = '/home/f/projects/MotionPaper/trial/output/'
    root_output_dir = '/home/f/projects/MotionPaper/trial/output'
    df = collect_all_user_data(root_dir, ignore_index=False, make_anonymous=True)
    
    fig, ax = plt.subplots(2,2, figsize=(10,8))
    ax = ax.ravel()
    X = df[(df.Modality=='afd5') & (df.Trainmode==0)]

    a = sns.boxplot(ax=ax[0], x="Duration", y="GroundTruth", hue="Trainmode", data=df[(df.Modality=='afd5')].reset_index(), showfliers=False, palette = sns.color_palette("Paired"))
    b = sns.boxplot(ax=ax[1], x="Duration", y="GroundTruth", hue="Correct", data=X.reset_index(), showfliers=False, hue_order=[1,0])
    c = sns.boxplot(ax=ax[2], x="Duration", y="UserID", hue="Correct", data=X, showfliers=False, hue_order=[1,0])
    d = sns.barplot(ax=ax[3], x="UserID", y="Correct", data=X, edgecolor=".2", facecolor=(0.2,0.2,0.02,0.3))
    
    leg_handles = a.get_legend_handles_labels()[0]
    a.legend(leg_handles, ['Test', 'Train'], title='Set')


    leg_handles = b.get_legend_handles_labels()[0]
    b.legend(leg_handles, ['Correct', 'Wrong'], title='')

    leg_handles = c.get_legend_handles_labels()[0]
    c.legend(leg_handles, ['Correct', 'Wrong'], title='')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    per_user_results()