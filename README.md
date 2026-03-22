# perceptual-kNN

 The goal of this project is to design an algorithm that, based on the physical parameters of a percussion synthesizer, can generate new parameters that lead to a sound close to the reference sound as perceived by humans. This work follows on from the study by Han Han, Vincent Lostalen, and Mathieu Lagrange [1], which aimed to reverse the synthesizer's path: starting with a sound, find the physical parameters that allow it to be reproduced. A solution based on a neural network can only produce a single set of parameters for a given sound, which is the driving force behind this project: to overcome this limitation. We use the same physical percussion sound model synthesizer employed by these authors [2].

1. Han Han, Vincent Lostanlen et Mathieu Lagrange. Perceptual-Neural-Physical Sound Matching. 2023. arXiv : 2301.02886 [cs.SD]. url : https://arxiv.org/abs/2301.02886
2. Han Han et Vincent Lostanlen. Perceptual Neural Physical Sound Matching. https://github.com/lylyhan/perceptual_neural_physical. 2023. 

## Dependencies

We use uv to handle dependencies, if needed all the packages needed are in the pyproject.toml

## Pre-computing the (phi o g)(theta) 

To precompute the values needed for the KNN, simply run the precompute_S.py script.

Inside you can change the number of process and the batch size to fit your GPU and VRAM.

With the full parameters dataset (subdiv = 10) the precomputations dataset will be a bit more than 5go.

## Running the project 

The code to run is in the main notebook

## Hearing examples

Some audio examples can be heard here : https://leon-albert.github.io/perceptual-kNN/ 
