import copy

import numpy as np
from scipy.stats import skewnorm
from Bragg_mirror_sf import *
import random
import copy
class Diode:
    """
    A simplified Diode class only considering field in real space
    """
    def __init__(self, input):
        self.diode_response = input['response_curve']

        # initial diode position
        self.input = input
        self.x0 = input['position']['x']
        self.y0 = input['position']['y']
        self.r = input['position']['radius']
        self.d = input['position']['d']

        self.tstart = 0.0
        self.beam = None

        # record for diode signal roundtrip by roundtrip
        self.diode_time_record = np.array([])
        self.diode_signal_record = np.array([])



    def update_beam(self, beam):
        self.beam = beam

    def update_tstart(self, tstart):
        """
        update the start time stamp of the diode signal, for turn-by-trun
        :param tstart:
        :return:
        """
        self.tstart = tstart


    def get_diode_response(self, intensity, tsep):

        skew_num = self.diode_response['skew_num']
        noise_level = self.diode_response['noise_level']
        n_sample = self.diode_response['n_sample']

        x = np.linspace(-10, 10, n_sample)
        y = skewnorm.pdf(x, skew_num)
        y /= np.max(y)
        y *= intensity
        y += np.random.normal(0, noise_level * intensity, len(x))

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x *= tsep
        x += self.tstart

        return x, y

    def get_diode_intensity(self, x0 = None, y0 = None):
        assert self.beam is not None, 'No beam to get diode response!'

        field = self.beam.get_field()

        if not x0:
            x0 = self.x0
        if not y0:
            y0 = self.y0

        # through the iris

        xmesh, ymesh = self.beam.xmesh, self.beam.ymesh
        field[(xmesh - x0) ** 2 + (ymesh - y0) ** 2 > self.r ** 2] = 0
        # get final intensity
        intensity = np.sum(np.abs(field) ** 2) * self.beam.dx * self.beam.dy
        return intensity

    def record_diode_signal(self, tsep, x0=None, y0=None):
        # get the peak intensity at the diode
        intensity = self.get_diode_intensity(x0 = x0, y0 = y0)
        # get diode response
        time, signal = self.get_diode_response(intensity, tsep)
        # append the signal to the record
        self.diode_time_record = np.append(self.diode_time_record, time)
        self.diode_signal_record = np.append(self.diode_signal_record, signal)

        self.update_tstart(self.tstart + tsep)

    def reset(self):
        self.tstart = 0.0
        self.beam = None

        self.x0 = self.input['position']['x']
        self.y0 = self.input['position']['y']

        # record for diode signal roundtrip by roundtrip
        self.diode_time_record = np.array([])
        self.diode_signal_record = np.array([])


class Diode_Bragg(Diode):
    def __init__(self, diode_input, beam_input, crystal_input):
        super().__init__(diode_input)
        self.Eph = beam_input['Eph']
        self.xlamds = 1.239842e-06/self.Eph
        self.npadx = beam_input['npadx']
        self.thickness = crystal_input['thickness']


    def get_Bragg_response(self, d_theta):
        theta = np.pi/4 + d_theta + self.theta_x
        R0H, R00 = get_Bragg_mirror_response_theta(theta=theta, eph_0=self.Eph, d=self.thickness)
        return R0H, R00

    def update_beam(self, beam):
        self.beam = beam

        dx = np.mean(np.diff(self.beam.x))
        dy = np.mean(np.diff(self.beam.y))
        Dkx = 2 * np.pi / dx
        Dky = 2 * np.pi / dy

        field = self.beam.get_field()
        nx, ny = field.shape
        nx_padded = nx + 2 * self.npadx

        self.kx = Dkx / 2. * np.linspace(-1., 1., nx_padded)
        self.ky = Dkx / 2. * np.linspace(-1., 1., ny)
        self.k = 2 * np.pi / self.xlamds
        self.theta_x = self.kx / self.k
        self.theta_y = self.ky / self.k


        field_padded = np.pad(self.field, ((self.npadx, self.npadx), (0, 0)))
        self.fftfld = np.fft.fftshift(np.fft.fft2(field_padded), axes=(0, 1))

    def forward_diffraction(self, d_theta):
        R0H, R00 = self.get_Bragg_response(d_theta + self.beam.xp)
        return np.einsum('i,ij->ij', R00, self.fftfld)


    def drift(self, fftfld):
        Ldrift = self.d
        kx_mesh, ky_mesh = np.meshgrid(self.kx, self.ky, indexing='ij')
        H = np.exp(-1j * self.xlamds * Ldrift * (kx_mesh ** 2 + ky_mesh ** 2) / (4 * np.pi))
        return fftfld * H

    def get_diode_intensity(self, d_theta, x0 = None, y0 = None, jitter_on = True):

        # transmit the crystal
        fftfld_transmit = self.forward_diffraction(d_theta)
        #drift to diode
        fftfld_transmit = self.drift(fftfld_transmit)
        #ifft to real space
        fld_transmit = np.fft.ifft2(np.fft.ifftshift(fftfld_transmit), axes = (0,1))
        #unpad
        fld_transmit = fld_transmit[self.npadx:-self.npadx]
        # through the iris
        if not x0:
            x0 = self.x0
        if not y0:
            y0 = self.y0
        r = self.r
        xmesh, ymesh = self.beam.xmesh, self.beam.ymesh
        fld_transmit[(xmesh - x0) ** 2 + (ymesh - y0) ** 2 > r ** 2] = 0
        # get final intensity
        intensity = np.sum(np.abs(fld_transmit)**2)* self.beam.dx * self.beam.dy
        # add 100% noise
        if jitter_on:
            intensity *= random.random()

        return intensity

