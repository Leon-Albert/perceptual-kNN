from src.ftm import rectangular_drum
import numpy as np
import pandas as pd
import torch
import time
from rich.progress import track,Progress,BarColumn, TextColumn, SpinnerColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

def find_neighbour(DataFrame_path,ref_index_list,k,dist,show_progress=False):
    # Initialize 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.from_numpy(pd.read_csv(DataFrame_path, index_col=0).to_numpy()).to(device).to(float)
    datasize = data.size(dim=0)
    dist_i = torch.zeros(2,len(ref_index_list)*k).to(device)
    dist_v = torch.zeros(1,len(ref_index_list)*k).to(device)[0]

    # Build tensor of the reference points (for the candidates it's data)
    tensor_refs = torch.zeros(len(ref_index_list),data.size(dim=1)).to(device).to(float)
    for i in range(len(ref_index_list)):
        id = ref_index_list[i]
        tensor_refs[i,:] = data[id,:]

    # Find the distances for each t_ref to all the candidates. 
    progress = Progress(
            TextColumn("[MAIN] Iterating References"),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
    )
    if show_progress:
        iterator = progress.track(range(tensor_refs.size(dim=0)))
    else:
        iterator = range(tensor_refs.size(dim=0))

    with progress:
        for i_ref in iterator:
            dist_tensor = dist(data,tensor_refs[i_ref,:],show_progress=show_progress)

            # Add the k shortest distances
            dist_tensor_sorted,sort_indices = torch.sort(dist_tensor, stable=True)

            dist_i[0,k*i_ref:k*(i_ref+1)] = i_ref*torch.ones(1,k)
            dist_i[1,k*i_ref:k*(i_ref+1)] = sort_indices[0:k].to(device)

            dist_v[k*i_ref:k*(i_ref+1)] = dist_tensor_sorted[0:k]

            # Remove the ref point from the futur candidates 
            data = torch.cat((data[:i_ref],data[i_ref+1:]))

    return dist_i.to(int),dist_v












def find_neighbour_backup(dataFramePath,nb_neighbour,dist,return_time=False,show_progress=False):
    distance_calculation = 0
    node_exploration = 0
    total_time = time.time()
    
    data = pd.read_csv(dataFramePath)
    data_size = int(data.size/6)
    parameters_name = ["omega","tau","p","d","alpha"]
    
    #List initialization
    closest_neighbour =  [ [0]*len(parameters_name) for _ in range(nb_neighbour) ]
    smallest_distances = [np.inf for z in range(nb_neighbour)]
    
    #graph exploration
    if show_progress:
        iterator = range(data_size)
    else:
        iterator = range(data_size)

    for i in iterator:
        #Get phi of the neighbour
        parameterLine = data.iloc[[i]]
        theta = np.array([ parameterLine[parameters_name[k]].iloc[0] for k in range(len(parameters_name)) ])
        
        time1 = time.time()
        dist_n = dist(theta)
        distance_calculation += time.time() - time1
        time1 = time.time()
        #check if the neighbour is one of the closest
        if (dist_n < smallest_distances[-1]):
            #Find position
            founded = False
            for k in range(nb_neighbour-2,-1,-1):
                if (dist_n > smallest_distances[k] and not(founded)):
                    smallest_distances.insert(k+1,dist_n)
                    closest_neighbour.insert(k+1,theta)
                    #Delete the furthest neighbour
                    smallest_distances = smallest_distances[:-1]
                    closest_neighbour = closest_neighbour[:-1]
                    founded = True
            if (not(founded)):
                smallest_distances.insert(0,dist_n)
                closest_neighbour.insert(0,theta)
                #Delete the furthest neighbour
                smallest_distances = smallest_distances[:-1]
                closest_neighbour = closest_neighbour[:-1] 
        node_exploration += time.time() - time1
    total_time = time.time() - total_time
    if(return_time):
        return closest_neighbour,[distance_calculation,node_exploration]
    return closest_neighbour
