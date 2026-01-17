import torch
import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import product
from functools import partial
import torch.multiprocessing as mp

from rich.progress import (
    Progress, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeRemainingColumn, 
    MofNCompleteColumn,
    SpinnerColumn
)

# Ensure these paths/modules are available in your environment
from src.ftm import constants as FTM_constants
from src.phi import JTFS_forward
from src.ftm import rectangular_drum
from src.dataset_utils import theta_ds_create

# --- GLOBAL STORAGE FOR WORKERS ---
active_workers = None

def init_worker(counter):
    global active_workers
    active_workers = counter

def S_logic(theta, logscale_val):
    theta_cuda = theta.to("cuda", non_blocking=True)
    return JTFS_forward(rectangular_drum(theta_cuda, logscale_val, **FTM_constants))

def worker_wrapper(row, f):
    global active_workers
    with active_workers.get_lock():
        active_workers.value += 1
    try:
        result = f(row).cpu()
    finally:
        with active_workers.get_lock():
            active_workers.value -= 1
    return result

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- CONFIGURATION ---
    logscale = True
    bounds = [['omega', 'tau', 'p', 'd', 'alpha'], [(2.4, 3.8), (0.4, 3), (-5, -0.7), (-5, -0.5), (10e-05, 1)]]
    num_processes = 4
    write_batch_size = 5000  
    num_max_tensor = 100000   

    FolderPath = os.path.join(sys.path[0], "data", "precompute_S")
    ResultPath = os.path.join(FolderPath, "S_dataset.parquet")
    DatasetPath = os.path.join(FolderPath, "param_dataset.csv")

    if not os.path.exists(FolderPath): os.makedirs(FolderPath)
    if os.path.exists(ResultPath): os.remove(ResultPath)

    DF_in_memory = theta_ds_create(bounds=bounds, subdiv=10, path=DatasetPath, write=True)
    rows_to_process = [torch.from_numpy(row).to(torch.float) for row in DF_in_memory.iloc[:num_max_tensor].values]

    shared_counter = mp.Value('i', 0)
    S_bound = partial(S_logic, logscale_val=logscale)
    worker = partial(worker_wrapper, f=S_bound)

    results_buffer = []
    parquet_writer = None 
    total_written = 0  # Tracker for the row_id column

    def get_speed(task):
        return f"{task.speed:.2f}" if task.speed else "0.00"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[yellow]{task.fields[fps]} it/s"), 
        TimeRemainingColumn(),
        TextColumn("[blue]Workers: {task.fields[workers]}"),
        TextColumn("{task.fields[status]}"),
        refresh_per_second=20
    ) as progress:

        task_id = progress.add_task("Computing", total=len(rows_to_process), fps="0.00", workers=0, status="")

        with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(shared_counter,)) as pool:
            for result in pool.imap(worker, rows_to_process):
                results_buffer.append(result)
                
                current_fps = get_speed(progress.tasks[task_id])
                progress.update(
                    task_id, 
                    advance=1, 
                    fps=current_fps, 
                    workers=shared_counter.value,
                    status=f"[cyan]({len(results_buffer)}/{write_batch_size})"
                )

                if len(results_buffer) >= write_batch_size:
                    progress.update(task_id, description="[bold red]Fast Save", status="[red]DISK I/O")
                    progress.refresh()
                    
                    # Convert buffer to DataFrame
                    chunk_df = pd.DataFrame(torch.stack(results_buffer).numpy())
                    chunk_df.columns = chunk_df.columns.astype(str) 
                    
                    # Add physical ID column for indexing
                    chunk_df['row_id'] = np.arange(total_written, total_written + len(chunk_df))
                    total_written += len(chunk_df)
                    
                    # Write to Parquet
                    table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(ResultPath, table.schema)
                    parquet_writer.write_table(table)
                    
                    results_buffer = []
                    torch.cuda.empty_cache()
                    
                    progress.update(task_id, description="Computing", status="")
                    progress.refresh()

        # Final Cleanup for remaining rows
        if results_buffer:
            progress.update(task_id, description="[bold green]Final Save", status="")
            chunk_df = pd.DataFrame(torch.stack(results_buffer).numpy())
            chunk_df.columns = chunk_df.columns.astype(str)
            chunk_df['row_id'] = np.arange(total_written, total_written + len(chunk_df))
            
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(ResultPath, table.schema)
            parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()

    print(f"Results saved to {ResultPath}")
    print(f"Total rows indexed: {total_written}")