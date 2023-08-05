from tools import isotime
import yaml
import numpy as np
from Bragg_mirror_sf import *
from scipy.stats import skewnorm
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
            self.thickness = 200e-6
            self.beam =  {'dgrid': 0.0001, 'ncar': 100, 'nslice': 50, 'w0': '40e-6', 'Eph': 9831.88, 'npadx': 1000}
            self.diode_response = {'type': 'function', 'skew_num': 10.0, 'noise_level': 0.05, 't_decay': 10e-9, 'n_sample': 1000}
            self.diode_position = {'d': 1.0, 'x': 0.0, 'y': 0.0, 'radius': 8e-05}

        self.init_beam()

    def parse_input(self, input_file):
        with open('../input/input.yaml') as f:
            input = yaml.load(f, Loader=yaml.FullLoader)
        self.check_input_consistency(input)
        self.input = input
        self.theta0 = input['crystal']['theta']/180*np.pi
        self.thickness = input['crystal']['d']
        self.beam = input['beam']
        self.diode_response = input['diode']['response_curve']
        self.diode_position = input['diode']['position']



    def check_input_consistency(self, input):
        self.required_inputs = ['beam', 'diode', 'crystal']

        allowed_params = self.required_inputs + []
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to {self.__class__.__name__}.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in self.required_inputs:
            assert req in input, f'Required input parameter {req} to {self.__class__.__name__}.__init__(**kwargs) was not found.'

    def init_beam(self):
        """
        Generate beam in real and k space
        :return:
        """
        dgrid = self.beam['dgrid']
        ncar = self.beam['ncar']
        nslice = self.beam['nslice']
        w0 = self.beam['w0']
        Eph = self.beam['Eph']
        self.npadx = self.beam['npadx']
        self.xlamds = 1.23984198 / Eph * 1e-6

        self.x = np.linspace(-dgrid, dgrid, ncar)
        self.y = np.linspace(-dgrid, dgrid, ncar)
        dx = np.mean(np.diff(self.x))
        dy = np.mean(np.diff(self.y))
        xmesh, ymesh = np.meshgrid(self.x, self.y, indexing='ij')
        self.dx, self.dy = dx, dy

        sigma = w0/2

        self.field = 1. / (2. * np.pi * sigma ** 2) * np.exp(
            -((xmesh) ** 2. / (2. * sigma ** 2.) + (ymesh) ** 2. / (2. * sigma ** 2.)))

        self.field /= np.sqrt(np.sum(np.abs(self.field) ** 2) * dx * dy)

        Dkx = 2 * np.pi / dx
        Dky = 2 * np.pi / dy
        nx_padded = ncar + 2 * self.npadx
        self.kx = Dkx / 2. * np.linspace(-1., 1., nx_padded)
        self.ky = Dkx / 2. * np.linspace(-1., 1., ncar)
        self.k = 2*np.pi/self.xlamds
        self.theta_x = self.kx/self.k
        self.theta_y = self.ky/self.k
        # pad in x
        field_padded = np.pad(self.field, ((self.npadx, self.npadx), (0, 0)))

        # fft in x and y
        self.fftfld = np.fft.fftshift(np.fft.fft2(field_padded), axes=(0, 1))


    def get_Bragg_response(self, d_theta):
        theta = self.theta0 + d_theta  + self.theta_x
        Eph = self.beam['Eph']
        d = self.thickness
        R0H, R00 = get_Bragg_mirror_response_theta(theta = theta, eph_0=Eph, d=d)
        return R0H, R00

    def forward_diffraction(self, d_theta):
        R0H, R00 = self.get_Bragg_response(d_theta)
        return np.einsum('i,ij->ij', R00, self.fftfld)


    def drift(self, fftfld):
        Ldrift = self.diode_position['d']
        kx_mesh, ky_mesh = np.meshgrid(self.kx, self.ky, indexing='ij')
        H = np.exp(-1j * self.xlamds * Ldrift * (kx_mesh ** 2 + ky_mesh ** 2) / (4 * np.pi))
        return fftfld * H

    def get_diode_response(self, intensity):

        skew_num = self.diode_response['skew_num']
        noise_level = self.diode_response['noise_level']
        t_decay = self.diode_response['t_decay']
        n_sample = self.diode_response['n_sample']

        x = np.linspace(-10, 10, n_sample)
        y = skewnorm.pdf(x, skew_num)
        y /= np.max(y)
        y *= intensity
        y += np.random.normal(0, noise_level * intensity, len(x))
        x = (x - np.min(x)) / np.max(x)
        x *= t_decay

        return x, y

    def get_diode_signal(self, d_theta):

        # transmit the crystal
        fftfld_transmit = self.forward_diffraction(d_theta)
        #drift to diode
        fftfld_transmit = self.drift(fftfld_transmit)
        #ifft to real space
        fld_transmit = np.fft.ifft2(np.fft.ifftshift(fftfld_transmit), axes = (0,1))
        #unpad
        fld_transmit = fld_transmit[self.npadx:-self.npadx]
        # through the iris
        x0 = self.diode_position['x']
        y0 = self.diode_position['y']
        r = self.diode_position['radius']
        xmesh, ymesh = np.meshgrid(self.x, self.y, indexing='ij')
        fld_transmit[(xmesh - x0) ** 2 + (ymesh - y0) ** 2 > r ** 2] = 0
        # get final intensity
        intensity = np.sum(np.abs(fld_transmit)**2)*self.dx*self.dy
        # get diode response
        time, signal = self.get_diode_response(intensity)
        return intensity, time, signal

    def scan_theta(self, d_theta_range):
        intensity_record = np.zeros(d_theta_range.shape)
        time_record = np.zeros((len(d_theta_range), self.diode_response['n_sample']))
        signal_record = np.zeros((len(d_theta_range), self.diode_response['n_sample']))

        for count, d_theta in enumerate(d_theta_range):
            intensity, time, signal = self.get_diode_signal(d_theta)
            intensity_record[count] = intensity
            time_record[count,:] = time
            signal_record[count,:] = signal

        return intensity_record, time_record, signal_record







