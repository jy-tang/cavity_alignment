import numpy as np
def make_gaus_beam(ncar, dgrid=400.e-6, w0=40.e-6, Dt=1e-6 / 3e8, t0=0., nslice=100, trms=2e-6 / 3e8, energy = 10e-6):
    xs = np.linspace(-1, 1, ncar) * dgrid
    ys = xs
    xv, yv = np.meshgrid(xs, ys, indexing = 'ij')

    sigx2 = (w0 / 2.) ** 2;
    fld = np.exp(-0.25 * (xv ** 2 + yv ** 2) / sigx2)

    ts = np.linspace(0, Dt, nslice)
    ts -= np.mean(ts)
    amps = np.exp(-0.25 * ((ts - t0) / trms) ** 2)

    print(trms)

    fld0 = np.zeros([nslice, ncar, ncar]) + 1j * 0.

    for ia, a in enumerate(amps):
        fld0[ia] = a * fld

    print(np.sum(np.abs(fld0)**2))
    print((np.sum(np.abs(fld0)**2)*Dt/nslice))
    fld0 *= energy/(np.sum(np.abs(fld0)**2)*Dt/nslice)

    return fld0