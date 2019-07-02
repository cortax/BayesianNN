from pandas import read_csv
from os import listdir


def load_dfs(path_name):

    assert(isinstance(path_name, str))

    if path_name[-1] == '/':
        path_name = path_name[:-1]

    file_names = listdir(path_name)

    dfs = []
    for file_name in file_names:
        if file_name.endswith('.csv'):
            dfs += [read_csv(path_name + '/' + file_name, skiprows=2)]

    return dfs





