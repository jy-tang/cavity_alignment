from cavity_beam_profiler import *
from tools import plot_2D_contour
import matplotlib.pyplot as plt

CF = cavity_profiler('input/input.yaml')

CF.insert_screen('x31')
CF.recirculate(dtheta2_x= -50e-6, use_diodeC2=True)