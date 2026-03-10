import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from itertools import product
import torch
import tqdm

"""Parameters Dataset"""

def theta_ds_create(bounds, subdiv, path, write=False):
    Dbase = np.zeros((subdiv, 4))
    for i in range(4):
        Dbase[:, i] = np.linspace(bounds[1][i][0], bounds[1][i][1], subdiv)
    baseDF = pd.DataFrame(data=Dbase, columns=bounds[0])
    D = list(product(baseDF['tau'], baseDF['p'], baseDF['d'], baseDF['alpha']))
    DF = pd.DataFrame(data=D, columns=bounds[0])
    if write:
        DF.to_csv(path, index=False)
    return DF


def S_ds_compute(DF, row_indices, S):

    if torch.is_tensor(row_indices):
        requested_pos = row_indices.detach().cpu().tolist()
    else:
        requested_pos = list(row_indices)

    all_processed_tensors = []

    for pos in tqdm.tqdm(requested_pos, desc="Computing S", leave=True):
            all_processed_tensors.append(S(DF[pos,:]))

    return torch.stack(all_processed_tensors)


"""S(theta) Dataset"""

def S_ds_read_given_rows(path, row_indices):

    if torch.is_tensor(row_indices):
        requested_ids = row_indices.detach().cpu().numpy()
    elif isinstance(row_indices, np.ndarray):
        requested_ids = row_indices
    else:
        requested_ids = np.array(row_indices)

    # We use ds.dataset to avoid loading the 50GB file metadata into RAM
    dataset = ds.dataset(path, format="parquet")
    
    scanner = dataset.scanner(
        filter=ds.field("row_id").isin(requested_ids.tolist()),
        columns=None  
    )

    # We keep the 'row_id' column for now so we can reorder correctly
    table = scanner.to_table()
    if table.num_rows == 0:
        return None

    # Reorder the S to match the row_indices order
    found_ids = table.column("row_id").to_numpy()
    id_to_idx = {id_val: i for i, id_val in enumerate(found_ids)}
    reorder_map = [id_to_idx[rid] for rid in requested_ids if rid in id_to_idx]
    
    # To numpy
    data_table = table.drop(["row_id"])
    raw_data = np.column_stack([col.to_numpy() for col in data_table.columns])
    
    ordered_data = raw_data[reorder_map]

    return torch.from_numpy(ordered_data)


def S_ds_read_given_rows_batch(path, row_indices, batch_size):

    total_indices = len(row_indices)
    collected_tensors = []

    for i in tqdm.tqdm(range(0, total_indices, batch_size), desc="Reading S", leave=True):
        batch_indices = row_indices[i : i + batch_size]

        batch_tensor = S_ds_read_given_rows(path, batch_indices)
        if batch_tensor is not None:
            collected_tensors.append(batch_tensor)

    return torch.cat(collected_tensors, dim=0)

