# The goal  of this script is to test the method with a large number of points, save the results in a svg file and store the calculation time

#===== Imports
import torch
import numpy as np
import pandas as pd
from itertools import product
import functools
# from perceptual-kNN.src.forward import *
# from perceptual-kNN.src.knn import *
# from perceptual-kNN.src.ftm import constants as FTM_constants
from tqdm import tqdm
import json
import time
#python program to check if a directory exists
import os

directory = "test_data/"
# Check whether the specified path exists or not
isExist = os.path.exists(directory[:-1])
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(directory[:-1])
   print("The new directory is created!")
else:
   print("Directory already exists!")

device = "cuda" if torch.cuda.is_available() else "cpu"

#===== Define the list of tests

dataset_parameters = [  {"bounds": [['omega', 'tau', 'p', 'd', 'alpha'],
                        [(2.400247964468862, 3.798136579655672),
                        (0.0700188044714488, 0.7999966616122908),
                        (-4.999978530884291, -0.6989804486272966),
                        (-4.99983759075039, -0.5229983775344527),
                        (1.2362882382361523e-05, 0.9999649724709304)]],
                         "subdiv":2,
                         "nbNeighbor":100,
                         "theta_ref_index":0
                        },
                         {"bounds": [['omega', 'tau', 'p', 'd', 'alpha'],
                        [(2.400247964468862, 3.798136579655672),
                        (0.0700188044714488, 0.7999966616122908),
                        (-4.999978530884291, -0.6989804486272966),
                        (-4.99983759075039, -0.5229983775344527),
                        (1.2362882382361523e-05, 0.9999649724709304)]],
                         "subdiv":4,
                         "nbNeighbor":100,
                         "theta_ref_index":0
                        },
                         {"bounds": [['omega', 'tau', 'p', 'd', 'alpha'],
                        [(2.400247964468862, 3.798136579655672),
                        (0.0700188044714488, 0.7999966616122908),
                        (-4.999978530884291, -0.6989804486272966),
                        (-4.99983759075039, -0.5229983775344527),
                        (1.2362882382361523e-05, 0.9999649724709304)]],
                         "subdiv":5,
                         "nbNeighbor":100,
                         "theta_ref_index":50
                        } ]

#===== Define all functions

#=== Synth algorithm

FTM_constants = {
    "x1": 0.4,
    "x2": 0.4,
    "h": 0.03,
    "l0": np.pi,
    "m1": 10,
    "m2": 10,
    "sr": 22050,
    "dur":2**16
}


def rectangular_drum(theta, logscale, **constants):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w11 = 10 ** theta[0] if logscale else theta[0]
    p = 10 ** theta[2] if logscale else theta[2]
    D = 10 ** theta[3] if logscale else theta[3]
    #theta
    tau11 = theta[1]
    alpha_side = theta[4]
    l0 = torch.tensor(constants['l0']).to(device)

    l2 = l0 * alpha_side 
    pi = torch.tensor(np.pi, dtype=torch.float64).to(device)

    beta_side = alpha_side + 1 / alpha_side
    S = l0 / pi * ((D * w11 * alpha_side)**2 + (p * alpha_side / tau11)**2)**0.25
    c_sq = (
        alpha_side * (1 / beta_side - p**2 * beta_side) / tau11**2 
        + alpha_side * w11**2 * (1 / beta_side - D**2 * beta_side)
    ) * (l0 / np.pi)**2
    T = c_sq # scalar
    d1 = 2 * (1 - p * beta_side) / tau11
    d3 = -2 * p * alpha_side / tau11 * (l0 / pi) **2 

    EI = S ** 4 

    mu = torch.arange(1, constants['m1'] + 1).to(device) #(m1,)
    mu2 = torch.arange(1, constants['m2'] + 1).to(device) #(m2,)
    dur = constants['dur']
    
    n = (mu[:,None] * pi / l0) ** 2 + (mu2[None,:] * pi / l2)**2 #(m1,m2)
    n2 = n ** 2 
    K = torch.sin(mu[:,None] * pi * constants['x1']) * torch.sin(mu2[None,:] * pi * constants['x2']) #(m1,m2)

    beta = EI * n2 + T * n #(m1, m2)
    alpha = (d1 - d3 * n)/2 # nonlinear
    omega = torch.sqrt(torch.abs(beta - alpha**2))

    #adaptively change mode number according to nyquist frequency
    mode_rejected = (omega / 2 / pi) > constants['sr'] / 2
    mode1_corr = constants['m1'] - max(torch.sum(mode_rejected, dim=0)) if constants['m1']-max(torch.sum(mode_rejected, dim=0))!=0 else constants['m1']
    mode2_corr = constants['m2'] - max(torch.sum(mode_rejected, dim=1)) if constants['m2']-max(torch.sum(mode_rejected, dim=1))!=0 else constants['m2']
    N = l0 * l2 / 4
    yi = (
        constants['h'] 
        * torch.sin(mu[:, None] * pi * constants['x1']) 
        * torch.sin(mu2[None, :] * pi * constants['x2']) 
        / omega #(m1, m2)
    ) 

    time_steps = torch.linspace(0, dur, dur).to(device) / constants['sr'] #(T,)
    y = torch.exp(-alpha[:,:,None] * time_steps[None, None, :]) * torch.sin(
        omega[:,:,None] * time_steps[None,None,:]
    ) # (m1, m2, T)

    y = yi[:,:,None] * y #(m1, m2, T)
    y_full = y * K[:,:,None] / N
    #mode_rejected = mode_rejected.unsqueeze(2).repeat(1,1,y_full.shape[-1])
    y_full = y_full[:mode1_corr, :mode2_corr, :]
    #y_full[mode_rejected] -= y_full[mode_rejected]
    y = torch.sum(y_full, dim=(0,1)) #(T,)
    y = y / torch.max(torch.abs(y))

    return y

#=== Phi definition
class FIRFilter(torch.nn.Module):
    
    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):

        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        import scipy.signal

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

        self.fir.weight.data = self.fir.weight.data.to(device)

    def forward(self, input):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        input = torch.nn.functional.conv1d(
            input.unsqueeze(0).to(torch.float), self.fir.weight.data, padding=self.ntaps // 2
        )
        return input.squeeze(0).to(torch.float)

#=== Naive k neighbours
def dist_naive_factory(theta_ref,phi,logscale):
    #calculation of the audio for the reference node
    audio_ref = rectangular_drum(theta_ref, logscale=logscale,**FTM_constants)
    phi_ref = phi(audio_ref)
    def naiveDistFunction(theta):
        audio_node = rectangular_drum(theta, logscale=logscale,**FTM_constants)
        phi_node = phi(audio_node)
        return (torch.sum(torch.pow(torch.subtract(phi_ref, phi_node), 2), dim=0)).cpu().detach().numpy()
    return naiveDistFunction
#=== Approximated k neighbours search
# First we need to create the M matrix

# M(theta0) = grad(Phi o g)(theta0).T * grad(Phi o g)(theta0)
# This return M = f(theta0)

def M_from_G(G):
    return torch.matmul(torch.transpose(G,0,1),G)

def M_from_theta(theta, G):
    return M_from_G(G(inputs=theta))

def x_from_theta(theta, logscale):
    """Compute x = g(theta)"""
    x = rectangular_drum(theta, logscale, **FTM_constants)
    return x

def pknn_forward(theta, logscale, Phi):
    """Compute S = (phi o g)(theta)"""
    g = functools.partial(x_from_theta,logscale=logscale)
    x = g(theta)
    S = Phi(x)
    return S

def pknn_forward_factory(logscale, Phi):
    """Compute S = (phi o g)""" 
    return functools.partial(pknn_forward, logscale=logscale, Phi=Phi)        


def M_factory(logscale,Phi):
    S_from_theta = pknn_forward_factory(logscale,Phi)
    #G = torch.func.jacfwd(S_from_theta)
    G = functools.partial(torch.autograd.functional.jacobian, func=S_from_theta, create_graph=False,strategy="forward-mode",vectorize=True)
    M = functools.partial(M_from_theta,G=G)
    return M

def dist_from_M_and_theta0(t_candidat, t_ref, M):
    return np.matmul(np.matmul(np.transpose(t_ref-t_candidat),M.cpu().detach().numpy()),t_ref-t_candidat)

def dist_approximated_factory(t_ref,Phi,logscale):
    M = M_factory(logscale,Phi)
    M = M(torch.tensor(t_ref, requires_grad=True).to(device))
    return functools.partial(dist_from_M_and_theta0, M=M, t_ref=t_ref)

#=== Create Parameters Dataset
# Create DataFrame and write it to a CSV file for later use

def create_DF(bounds, subdiv, path):
    
    #Linspace of every parameters of size k
    Dbase = np.zeros((subdiv,5))
    for i in range(5):
        Dbase[:,i] = np.linspace(bounds[1][i][0],bounds[1][i][1],subdiv)
    baseDF = pd.DataFrame(data=Dbase,columns=bounds[0])

    #Product of the linspaces to get all the possible combinations (size subdiv**5, will take time)
    D = list(product(baseDF['omega'],baseDF['tau'],baseDF['p'],baseDF['d'],baseDF['alpha']))
    DF = pd.DataFrame(data=D,columns=bounds[0])

    DF.to_csv(path)
    
    return DF

print("_"*20)
print("creation of the dataset")
for test_id in range(len(dataset_parameters)):
    print("    Dataset : " + str(test_id))
    path_name = directory + "test_" + str(test_id) + "_parameter_subdiv_" + str(dataset_parameters[test_id]["subdiv"])  + ".csv"
    create_DF(bounds=dataset_parameters[test_id]["bounds"], subdiv=dataset_parameters[test_id]["subdiv"], path=path_name)
    
print("_"*20)


#===== Knn search algorithm
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

#=== knn constants

parameters_name = ["omega","tau","p","d","alpha"]
logscale = True
Phi = FIRFilter()

#====== Run the list of tests
for testNumber in range(len(dataset_parameters)):
    print("******************************")
    print("Begining of the test number n°",testNumber)
    print("******************************")

    # Retrive information from the data set
    nbNeighbor = dataset_parameters[testNumber]["nbNeighbor"]
    theta_ref_index = dataset_parameters[testNumber]["theta_ref_index"]
    
    #Get the reference Theta
    print("Reading Dataset")
    DatasetPath = directory + "test_" + str(testNumber) + "_parameter_subdiv_" + str(dataset_parameters[testNumber]["subdiv"])  + ".csv"
    DS = pd.read_csv(DatasetPath, index_col=0)
    theta_ref_line = DS.iloc[[theta_ref_index]]
    theta_ref = [ theta_ref_line[parameters_name[k]].iloc[0] for k in range(len(parameters_name)) ]
    
    #Find the knn (Naive)
    print("Naive knn")
    
    dist = dist_naive_factory(theta_ref, Phi, logscale)
    naive_neighbours,calculation_time_naive = find_neighbour(DatasetPath,nbNeighbor,dist,return_time=True,show_progress=True)
    
    #Find the knn (Approx)
    print("Approx knn")
    
    dist = dist_approximated_factory(theta_ref, Phi, logscale)
    approx_neighbours,calculation_time_approx = find_neighbour(DatasetPath,nbNeighbor,dist,return_time=True,show_progress=True)

    #Save the data
    #Write the neighbours to a CSV file 
    print("Save the results :")
    print('Write naive resuts')
    path_name = directory + "test_" + str(testNumber) + "_naive_knn.csv"
    naive_DF = pd.DataFrame(np.array(naive_neighbours), columns=(parameters_name))
    naive_DF.to_csv(path_name)
    
    print('Write approx resuts')
    path_name = directory + "test_" + str(testNumber) + "_approx_knn.csv"
    approx_DF = pd.DataFrame(np.array(approx_neighbours), columns=(parameters_name))
    approx_DF.to_csv(path_name)

    #Add time calculation to the dictionary
    dataset_parameters[testNumber]["calculation_time_naive"] = { "distance_calculation" : calculation_time_naive[0],"node_exploration": calculation_time_naive[1] } 
    dataset_parameters[testNumber]["calculation_time_approx"] = { "distance_calculation" : calculation_time_approx[0],"node_exploration": calculation_time_approx[1] } 


#===== Store every parameter
for test_id in range(len(dataset_parameters)):
    filename = directory + "test_" + str(test_id) + "_test_parameters.json"
    with open(filename, "w") as f:
        json.dump(dataset_parameters[test_id], f)