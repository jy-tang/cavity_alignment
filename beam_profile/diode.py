import numpy as np
from scipy.stats import skewnorm
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
        #y += np.random.normal(0, noise_level * intensity, len(x))

        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        x *= tsep
        x += self.tstart

        return x, y



    def record_diode_signal(self, tsep, x0=None, y0=None):

        assert self.beam is not None, 'No beam to get diode response!'

        field = self.beam.get_field()

        if not x0:
            x0 = self.x0
        if not y0:
            y0 = self.y0


        # through the iris

        xmesh, ymesh = np.meshgrid(self.beam.x, self.beam.y, indexing='ij')
        field[(xmesh - x0) ** 2 + (ymesh - y0) ** 2 > self.r ** 2] = 0
        # get final intensity
        intensity = np.sum(np.abs(field) ** 2) * self.beam.dx * self.beam.dy
        # get diode response
        time, signal = self.get_diode_response(intensity, tsep)

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