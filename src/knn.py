import time
from src.ftm import rectangular_drum
import numpy as np
import pandas as pd

def find_neighbour(dataFramePath,nb_neighbour,dist,return_time=False):
    distance_calculation = 0
    node_exploration = 0
    total_time = time.time()
    constants = {
        "x1": 0.4,
        "x2": 0.4,
        "h": 0.03,
        "l0": np.pi,
        "m1": 10,
        "m2": 10,
        "sr": 22050,
        "dur":2**16
    }
    
    data = pd.read_csv(dataFramePath)
    data_size = int(data.size/6)
    parameters_name = ["omega","tau","p","d","alpha"]
    
    #List initialization
    closest_neighbour =  [[0]*len(parameters_name)]*nb_neighbour
    smallest_distances = [np.inf for z in range(nb_neighbour)]
    
    #graph exploration
    for i in range(data_size):
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
            for k in range(nb_neighbour-2,0,-1):
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

# # Example of usage
#theta5 = [2.448304103287737,0.6724932913451673,-1.4882183960726143,-1.1237355795715704,0.9775323978804632]
#theta6= [2.559894429686223,0.5765175855542937,-1.020964798228077,-0.1456230260198473,0.6825837860862676]
# def createNaiveDist(theta_ref,phi):
#     #calculation of the audio for the reference node
#     audio_ref = rectangular_drum(theta_ref, True,**constants)
#     phi_ref = phi(audio_ref)
#     def naiveDistFunction(theta):
#         audio_node = rectangular_drum(theta, True,**constants)
#         phi_node = phi(audio_node)
#         return torch.sqrt(torch.sum(torch.pow(torch.subtract(phi_ref, phi_node), 2), dim=0))
#     return naiveDistFunction

# def phi_test(x):
#     return x
    
# dist_test = createNaiveDist(theta6,phi_test)
# thetaList = find_neighbour('full_param_log.csv',20,dist_test,False)