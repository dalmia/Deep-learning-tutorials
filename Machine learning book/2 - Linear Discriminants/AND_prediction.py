import numpy as np
import pcn

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
targets = np.array([[0], [0], [0], [1]])
p = pcn.pcn(inputs, targets)
p.pcntrain(inputs, targets, 0.25, 6)
