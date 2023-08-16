import copy

import yaml
import numpy as np
from scipy.stats import skewnorm
import random
from wavefront import GaussianWavefront
import copy
class cavity_profiler:
    """
    The main class of a fast diode simulator
    """
    def __init__(self, input_file):

        self.parse_input(input_file)
        self.input_file = input_file

        self.init_beam()


    def parse_input(self, input_file):
        with open(input_file) as f:
            input = yaml.load(f, Loader=yaml.FullLoader)
        self.check_input_consistency(input)
        self.input = input
        self.crystal_h = input['crystal']['h']
        self.beam0 = input['beam']

        self.L1 = input['cavity_conf']['long_arm']
        self.L2 = input['cavity_conf']['short_arm']
        self.f = input['cavity_conf']['focal_length']

        self.screens = input['screens']

    def init_beam(self):
        self.beam = GaussianWavefront(self.beam0)
        self.record = {}



    def check_input_consistency(self, input):
        self.required_inputs = ['crystal', 'beam', 'cavity_conf', 'screens']

        allowed_params = self.required_inputs + []
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'


    def recirculate(self, dtheta1_x = 0.0, dtheta1_y = 0.0,
                          dtheta2_x = 0.0, dtheta2_y = 0.0,
                          dtheta3_x = 0.0, dtheta3_y = 0.0,
                          dtheta4_x = 0.0, dtheta4_y = 0.0,
                          dx_CRL1 = 0.0, dx_CRL2 = 0.0):
        # from the center of the undulator to x11
        Ldrift = self.screens['x11'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x11'] = copy.deepcopy(self.beam)

        # from x11 to C1
        Ldrift = self.L1/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C1
        self.beam.crystal_mirror(h = self.crystal_h, dtheta_x = dtheta1_x, dtheta_y = dtheta1_y)

        # from C1 to x12
        Ldrift = self.screens['x12'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x12'] = copy.deepcopy(self.beam)

        # from x12 to CRL1
        Ldrift = self.L1/2 + self.L2/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # CRL1
        self.beam.focal_lens(f = self.f, delta_x = dx_CRL1)

        # CRL1 to x21
        Ldrift = self.screens['x21'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x21'] = copy.deepcopy(self.beam)

        # x21 to C2
        Ldrift = self.L2 + self.L1/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C2
        self.beam.crystal_mirror(h=self.crystal_h, dtheta_x=dtheta2_x, dtheta_y=dtheta2_y)

        # C2 to x22
        Ldrift = self.screens['x22'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x22'] = copy.deepcopy(self.beam)

        # x22 to x31
        Ldrift = self.screens['x31'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x31'] = copy.deepcopy(self.beam)

        # x31 to C3
        Ldrift = self.L1/2 + self.L2 + self.L1 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C3
        self.beam.crystal_mirror(h=self.crystal_h, dtheta_x=dtheta3_x, dtheta_y=dtheta3_y)

        # C3 to x32
        Ldrift = self.screens['x32'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x32'] = copy.deepcopy(self.beam)

        # x32 to CRL2
        Ldrift = self.L1/2 + self.L2 + self.L1 + self.L2/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # CRL2
        self.beam.focal_lens(f=self.f, delta_x=dx_CRL2)

        # CRL2 to x41
        Ldrift = self.screens['x41'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x41'] = copy.deepcopy(self.beam)

        # x41 to C4
        Ldrift = self.L1/2 + self.L2 + self.L1 + self.L2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C4
        self.beam.crystal_mirror(h=self.crystal_h, dtheta_x=dtheta4_x, dtheta_y=dtheta4_y)

        # C4 to x42
        Ldrift = self.screens['x42'] - self.beam.z_proj
        self.beam.propagate(Ldrift)
        self.record['x42'] = copy.deepcopy(self.beam)

        # x41 to the undulator center
        Ldrift = (self.L1 + self.L2)*2 - self.beam.z_proj
        self.beam.propagate(Ldrift)


    def get_profile(self, screen_name):
        assert screen_name in self.record.keys(), "Error, no such screen!"
        bt = self.record[screen_name]
        return bt.x, bt.y, bt.get_field()
