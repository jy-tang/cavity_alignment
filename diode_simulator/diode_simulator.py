from tools import isotime
import yaml
import numpy as np
class Diode:
    """
    The main class of a fast diode simulator
    """

    def __init__(self, input_file=None):

        self.timestamp = isotime()
        if input_file:
            self.parse_input(input_file)
            self.input_file = input_file

        else:   #default number
            self.theta0 = np.pi/4
            self.beam =  {'sigma_x': 2e-05, 'sigma_y': 2e-05, 'peak_power': 1e-08, 'nx': 100, 'ny': 100, 'napdx': 1000}
            self.diode = {'response_curve': {'type': 'function', 'skew_num': 1.0, 'noise_level': 0.1},
                          'position': {'x': 0.0, 'y': 0.0, 'radius': 8e-05}}





    def parse_input(self, input_file):
        with open('../input/input.yaml') as f:
            input = yaml.load(f, Loader=yaml.FullLoader)
        self.check_input_consistency(input)
        self.input = input
        self.theta0 = input['crystal']['theta']/180*np.pi
        self.beam = input['beam']
        self.diode = input['diode']


    def check_input_consistency(self, input):
        self.required_inputs = ['beam', 'diode', 'crystal', 'scan']

        allowed_params = self.required_inputs + []
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def initialized_beam(self):
        sigma_x = self.beam['sigma_x']
        sigma_y = self.beam['sigma_y']

