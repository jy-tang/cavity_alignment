from cavity_beam_profiler import *
from tools import plot_2D_contour
from matplotlib import cm
import matplotlib.pyplot as plt

CF = cavity_profiler('./input/input.yaml')
CF.recirculate(dtheta1_x= 0.0)
x, y, p = CF.get_profile(screen_name = 'x42')
plot_2D_contour(x, y, np.abs(p)**2)