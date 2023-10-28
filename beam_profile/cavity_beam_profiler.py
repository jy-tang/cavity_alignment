import copy

import yaml
import numpy as np
import random
from wavefront import GaussianWavefront
import copy
from diode import *
class cavity_profiler:
    """
    The main class of a cavity simulator
    """
    def __init__(self, input_file):

        self.parse_input(input_file)
        self.input_file = input_file

        self.init_beam()

        self.screen_in = None           # The screen inserted
        self.stop_flag = False           # flags showing the beam is blocked by a screen


    def parse_input(self, input_file):
        with open(input_file) as f:
            input = yaml.load(f, Loader=yaml.FullLoader)
        self.check_input_consistency(input)
        self.input = input
        self.crystal_h = input['crystal']['h']
        self.crystal_R0  = input['crystal']['R0']
        self.beam0 = input['beam']

        self.L1 = input['cavity_conf']['long_arm']
        self.L2 = input['cavity_conf']['short_arm']
        self.f = input['cavity_conf']['focal_length']
        self.roundtrip_time = (self.L1 + self.L2)*2/299792458

        self.screens = input['screens']
        self.trans_screens = input['trans_screens']

        # Diode E
        self.diode_E = Diode(input['diode_E'])

        # Diode C2
        self.diode_C2 = Diode_Bragg(input['diode_C2'], input['beam'], input['crystal'])

        #Initialize record
        self.record = {}
        nx = ny = self.beam0['ncar']
        for screen in self.screens:
            self.record[screen] = np.zeros((nx, ny))
        for screen in self.trans_screens:
            self.record[screen] = np.zeros((nx, ny))




    def init_beam(self):
        self.beam = GaussianWavefront(self.beam0)


    def insert_screen(self, screen_name):
        self.screen_in = screen_name

    def remove_screen(self):
        self.screen_in = None

    def reset(self):
        """
        reset everything to its initial condition
        :return:
        """
        self.beam.reset()

        #self.record = {}
        nx = ny = self.beam0['ncar']
        for screen in self.record:
            self.record[screen] = np.zeros((nx, ny))
        self.diode_E.reset()
        self.diode_C2.reset()
        #self.screen_in = None
        self.stop_flag = False


    def check_input_consistency(self, input):
        self.required_inputs = ['crystal', 'beam', 'cavity_conf', 'screens', 'diode_E', 'trans_screens', 'diode_C2']

        allowed_params = self.required_inputs + []
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def propagate_and_record_screen(self, current_screen):
        if current_screen in self.trans_screens:  # is a transmissive screen, add up the profile from each turn
            Ldrift = self.trans_screens[current_screen]['position'] - self.beam.z_proj
        else:
            Ldrift = self.screens[current_screen] - self.beam.z_proj
        self.beam.propagate(Ldrift)


        if self.screen_in == current_screen:   # if the screen is in
            if current_screen in self.trans_screens:   # is a transmissive screen, add up the profile from each turn
                transmission = self.trans_screens[current_screen]['transmission']
                self.beam.A *= np.sqrt(transmission)
                self.record[current_screen] += np.abs(self.beam.get_field()) ** 2
            else:  # else, block the beam from the screen
                self.record[current_screen] = np.abs(self.beam.get_field()) ** 2
                self.stop_flag = True

    def recirculate(self, dtheta1_x = 0.0, dtheta1_y = 0.0,
                          dtheta2_x = 0.0, dtheta2_y = 0.0,
                          dtheta3_x = 0.0, dtheta3_y = 0.0,
                          dtheta4_x = 0.0, dtheta4_y = 0.0,
                          dx_CRL1 = 0.0, dx_CRL2 = 0.0,
                          dy_CRL1 = 0.0, dy_CRL2 = 0.0,
                          dx_diodeE = 0.0, dy_diodeE = 0.0,
                          use_diodeC2 = False, dx_diodeC2 = 0.0, dy_diodeC2 = 0.0):

        if self.stop_flag:
            print('Beam blocked!')
            return
        # from the center of the undulator to x11
        current_screen = 'x11'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # from x11 to C1
        Ldrift = self.L1/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C1
        self.beam.crystal_mirror(h = self.crystal_h, R = self.crystal_R0, dtheta_x = dtheta1_x, dtheta_y = dtheta1_y)

        # from C1 to x10
        current_screen = 'x10'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # from x10 to CRL1
        Ldrift = self.L1/2 + self.L2/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # CRL1
        self.beam.focal_lens(f = self.f, delta_x = dx_CRL1, delta_y = dy_CRL1)

        # CRL1 to x21
        current_screen = 'x21'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # x21 to C2
        Ldrift = self.L2 + self.L1/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        if use_diodeC2:
            diffract_beam = copy.deepcopy(self.beam)

            self.diode_C2.update_beam(diffract_beam)
            self.diode_C2.record_diode_signal(tsep=self.roundtrip_time,
                                             x0=dx_diodeC2, y0=dy_diodeC2, d_theta= dtheta2_x)

        # reflect from C2
        self.beam.crystal_mirror(h=self.crystal_h, R = self.crystal_R0, dtheta_x=dtheta2_x, dtheta_y=dtheta2_y)



        # C2 to x23
        current_screen = 'x23'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # from x23 to Station 3
        diffract_beam = copy.deepcopy(self.beam)
        Ldrift = self.L1/2 + self.L2 + self.L1  - diffract_beam.z_proj + self.diode_E.d
        diffract_beam.propagate(Ldrift)

        self.diode_E.update_beam(diffract_beam)
        self.diode_E.record_diode_signal(tsep = self.roundtrip_time,
                                         x0 = dx_diodeE, y0 = dy_diodeE)

        # x23 to x24
        current_screen = 'x24'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # x24 to x31
        current_screen = 'x31'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # x31 to C3
        Ldrift = self.L1/2 + self.L2 + self.L1 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C3
        self.beam.crystal_mirror(h=self.crystal_h, R = self.crystal_R0, dtheta_x=dtheta3_x, dtheta_y=dtheta3_y)

        # C3 to x32
        #Ldrift = self.screens['x32'] - self.beam.z_proj
        #self.beam.propagate(Ldrift)
        #self.record['x32'] = copy.deepcopy(self.beam)

        # C3 to CRL2
        Ldrift = self.L1/2 + self.L2 + self.L1 + self.L2/2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # CRL2
        self.beam.focal_lens(f=self.f, delta_x=dx_CRL2, delta_y=dy_CRL2)

        # CRL2 to x41
        current_screen = 'x41'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # x41 to C4
        Ldrift = self.L1/2 + self.L2 + self.L1 + self.L2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # reflect from C4
        self.beam.crystal_mirror(h=self.crystal_h, R = self.crystal_R0, dtheta_x=dtheta4_x, dtheta_y=dtheta4_y)

        # C4 to x42
        current_screen = 'x42'
        self.propagate_and_record_screen(current_screen)
        if self.stop_flag:
            return

        # x42 to the undulator center
        Ldrift = (self.L1 + self.L2)*2 - self.beam.z_proj
        self.beam.propagate(Ldrift)

        # set z and z_proj to zero
        self.beam.z = 0.0
        self.beam.z_proj =0.0


    def get_profile(self, screen_name):
        assert screen_name in self.record.keys(), "Error, no such screen!"
        bt = self.record[screen_name]
        return self.beam.x, self.beam.y, bt


    def get_diodeE_signal(self):
        return self.diode_E.diode_time_record, self.diode_E.diode_signal_record
    def get_diodeC2_signal(self):
        return self.diode_E.diode_time_record, self.diode_E.diode_signal_record