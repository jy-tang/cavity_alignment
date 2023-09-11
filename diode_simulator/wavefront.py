import numpy as np

class GaussianWavefront:
    """
    generate a Gaussian wavefront
    z: current distance from the waist (R(z = 0)  = infinity)
    q0, q: complex beam parameter at the initial position(z = 0) and current positio
    w0, w: beam waist
    rho0, rho: 1/R(z), radius of curvature
    x, y: transverse grid
    zR: rayleigh length

    """

    def __init__(self, input):
        self.dgrid = input['dgrid']
        self.ncar = input['ncar']
        self.w = self.w0 = input['w0']
        self.Eph = input['Eph']
        self.xlamds = 1.23984198 /self.Eph * 1e-6
        self.k = 2*np.pi/self.xlamds
        self.zR = np.pi*self.w0**2/self.xlamds
        self.A = 1.0 # field amplitude

        self.z = 0.0    # initial position
        self.q = self.q0 = 1j*np.pi*self.w0**2/self.xlamds
        self.rho = self.rho0 = 0.0    #1/R(z = 0)

        self.x = self.y = np.linspace(-self.dgrid, self.dgrid, self.ncar)

        self.x0 = self.y0 = 0.0 # center of the beam
        self.xp = self.yp = 0.0   # angle of the central axis

        self.dx = self.dy = 2*self.dgrid/self.ncar
        self.xmesh, self.ymesh = np.meshgrid(self.x, self.y, indexing='ij')

    def get_field(self):
        return 1/self.q*np.exp(-1j*self.k*(self.xmesh**2 + self.ymesh**2)/2/self.q)

    def update(self):
        self.rho = np.real(1/self.q)
        self.w = np.sqrt(-self.xlamds/np.pi/np.imag(1/self.q))
    def propagation(self, Ldrift):
        """
        propagation in free space
        :param Ldrift: distance of the drift
        :return:
        """
        #d = Ldrift/np.cos(self.theta_x)/np.cos(self.theta_y)
        d = Ldrift
        self.z += d
        self.q += d
        self.x0 += Ldrift*self.xp
        self.y0 += Ldrift*self.yp
        self.update()

    def focal_lens(self, f, delta_x = 0):
        """
        go throught a focal lens
        :param f: focal length
        :param delta_x: lens displacement
        :return:
        """
        A = 1
        B = 0
        C = -1/f
        D = 1
        self.q = (A*self.q + B)/(C*self.q + D)
        # update waist, curvature in terms of q
        self.update()
        # update x xp
        self.xp += -1/f*self.x
        self.xp += delta_x/f



    def crystal_mirror(self, dtheta_x = 0, dtheta_y = 0):
        """
        R*exp(ih kx)
        :param dtheta:
        :return:
        """
        self.x *= -1
        self.xp *= -1
        self.xp += dtheta_x
        self.yp += dtheta_y















