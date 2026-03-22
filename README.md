# perceptual-kNN

Finding the perceptual neighbours of sound generated from a drum syntheziser, using different methods to approximate the distance in the perceptual domain

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
