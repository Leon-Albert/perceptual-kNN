import time
from src.ftm import rectangular_drum
import numpy as np
import pandas as pd
from tqdm import tqdm

def find_neighbour(dataFramePath,nb_neighbour,dist,return_time=False,show_progress=False):
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
        iterator = tqdm(range(data_size))
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