import os
import pandas as pd
import random


def get_usefull_data_from_df(data):
    predictions = set(data.Prediction)
    gtlabels = set(data.GroundTruth)
    labels = list(set.union(predictions, gtlabels))
    labels.sort()
    n_classes = len(labels)
    return predictions, gtlabels, labels, n_classes

def get_user_csv(root_dir, username):
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if name == f'{username}.csv':
                print(f'{name}: == {os.path.join(path, name)}')
                csv_file = os.path.join(path, name)
                return pd.read_csv(csv_file)
    raise ValueError("couldnt find user")


def collect_all_user_data(root_dir, ignore_index=True, make_anonymous=False):
    all_csv_files = []
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            if name.endswith('.csv'):
                csv_file = os.path.join(path, name)
                print(csv_file)
                all_csv_files.append(csv_file)
    df = pd.concat(map(pd.read_csv, all_csv_files), ignore_index=ignore_index)

    if make_anonymous == True:
        userlist = list(df['UserID'].unique())
        random.shuffle(userlist)
        for id, user in enumerate(userlist):
            newname = f'ID{id+1}'
            df['UserID'].replace(user, newname, inplace=True)

    return df