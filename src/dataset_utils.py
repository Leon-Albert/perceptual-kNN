import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from itertools import product
import torch

def theta_ds_create(bounds, subdiv, path, write=False):
    print('Generating parameter grid...')
    Dbase = np.zeros((subdiv, 5))
    for i in range(5):
        Dbase[:, i] = np.linspace(bounds[1][i][0], bounds[1][i][1], subdiv)
    baseDF = pd.DataFrame(data=Dbase, columns=bounds[0])
    D = list(product(baseDF['omega'], baseDF['tau'], baseDF['p'], baseDF['d'], baseDF['alpha']))
    DF = pd.DataFrame(data=D, columns=bounds[0])
    if write:
        DF.to_csv(path, index=False)
    return DF

def S_ds_read_single_row(path, row_idx):
    table = pq.read_table(
        path, 
        filters=[("row_id", "==", row_idx)]
    )
    if table.num_rows == 0:
        return None
    data_table = table.drop(["row_id"])
    return torch.from_numpy(data_table.to_pandas().values).flatten()


