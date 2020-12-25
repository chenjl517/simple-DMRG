import numpy as np

basicCell=np.array([
    [0.25,0,0,0],
    [0,-0.25,0.5,0],
    [0,0.5,-0.25,0],
    [0,0,0,0.25]
])

Sx=np.array([
    [0,0.5],
    [0.5,0]
])
Sy=np.array([
    [0,-1j/2],
    [1j/2,0]
])
Sz=np.array([
    [0.5,0],
    [0,-0.5]
])