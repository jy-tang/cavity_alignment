from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from Bragg_mirror_sf import *
def Bragg_mirror(photon_energies_eV, angles_rad, d=100e-6):
    e_charge = 1.602176565e-19  # charge unit[C]
    # h_Plank  = 6.62607004e-34;      # Plank constant [J-sec]
    h_Plank = 4.135667696e-15  # Plank constant [eV-sec]
    c_speed = 299792458  # speed of light[m/sec]
    pi = np.pi

    ####### Crystal electric susceptibility data from XOP ######
    chi_0 = -0.151228E-04 + 1j * 0.146454E-07  # Diamond electric susceptibility vacuum
    chi_h = 0.372253E-05 - 1j * 0.141091E-07
    chi_hbar = chi_h  # for centro-symmetric crystal
    ############################################################
    dh = 0.89170487e-10  # 0.891704865e-10; #interplanar distance (d-spacing?) (for 400 diamond at 9831 eV for 45 degree bragg?)

    # d = 100e-6; #crystal thickness

    eph = photon_energies_eV
    # eph = np.linspace(9831.65,9832.25,5e3+1); # photon energy in eV over which to evaluate the system
    lambd = h_Plank * c_speed / eph
    theta_0 = 45.0 * pi / 180.  # nominal angle of incidence, most likely equal to the Bragg angle for the target photon energy
    # dtheta_0 = 50e-6; # range of theta to interate over
    # ntheta_0 = 5e3+1;
    # theta = np.linspace(theta_0-dtheta_0,theta_0+dtheta_0,ntheta_0); # angle of incidence in radians over which to evaluate the system
    theta = angles_rad  # + theta_0 # make sure to center the angles on 45 deg

    Lambda, Theta = np.meshgrid(lambd, theta)

    eta = 0.0;  # asymetry angle
    gamma_0 = np.cos(Theta + eta - pi / 2.)
    gamma_h = np.cos(pi / 2. + Theta - eta)
    b = gamma_0 / gamma_h  # asymmetry factor
    P = 1.0;  # polarization factor, sigma case = 1

    H = 2. * pi / dh  # Bragg vector
    K0 = 2. * pi / Lambda  # magnitude of K0 vector
    # alpha = (2*K0*H*np.sin(Theta)-H**2)/(K0**2); #Deviation parameter
    alpha = (H / K0) * (H / K0 - 2. * gamma_0)

    lambda_hs = np.sin(Theta) / (K0 * np.abs(P) * np.sqrt(chi_h * chi_hbar))  # symmetric extinction length

    lambda_h = np.sqrt(gamma_0 * np.abs(gamma_h)) / (K0 * np.abs(P) * np.sqrt(chi_h * chi_hbar))  # extinction length

    y = K0 * lambda_h / (2. * gamma_0) * (b * alpha + chi_0 * (1. - b))

    Y1 = -y + np.sqrt(y ** 2 + b / np.abs(b))
    Y2 = -y - np.sqrt(y ** 2 + b / np.abs(b))
    G = np.sqrt(np.abs(b) * chi_h * chi_hbar) / chi_hbar
    R1 = G * Y1
    R2 = G * Y2
    A = d / lambda_h


    R0H = R1 * R2 * (1 - np.exp(1j * A / 2 * (Y1 - Y2))) / (
                    R2 - R1 * np.exp(1j * A / 2 * (Y1 - Y2)))  # bragg diffraction amplitude

    R00 = np.exp(1j * (chi_0 * K0 * d / 2 / gamma_0 + A / 2 * Y1)) * (R2 - R1) / (
                    R2 - R1 * np.exp(1j * A / 2 * (Y1 - Y2)))  # forward diffraction amplitude
        # C = np.exp(1j*chi_0*K0*d/(2*gamma_0)); #prompt response
        # R001 = R00-C;
    return R0H, R00


def plot_Bragg_mirror_response_slice(eph=None, theta=None, theta_slice=np.pi / 4., d=100e-6):
    pi = np.pi
    if eph is None:
        eph = np.linspace(9831, 9833, 5001)
    theta_0 = 45.0 * pi / 180.
    dtheta_0 = 50e-6
    ntheta_0 = 101
    # theta_0 *= 0.
    if not theta:
        theta = np.linspace(theta_0 - dtheta_0, theta_0 + dtheta_0, ntheta_0)

    R0H, R00 = Bragg_mirror(photon_energies_eV = eph, angles_rad = theta, d=d)

    Eph, Thetaurad = np.meshgrid(eph, 1e6 * (theta - theta_slice))
    cut = Thetaurad == 0;

    plt.figure(figsize=(10, 6))
    plt.plot(Eph[cut], np.abs(R0H[cut]) ** 2)
    plt.title(['Angle = 45 deg'])
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Bragg diffration intensity')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(Eph[cut], np.abs(R00[cut]) ** 2)
    plt.title(['Angle = 45 deg'])
    plt.xlabel('Photon energy (eV)')
    plt.ylabel('Forward diffration intensity')
    plt.show()

    return Eph[cut], np.abs(R0H[cut]) ** 2, np.abs(R00[cut]) ** 2


# plotting stuff (slow)
def plot_Bragg_mirror_response():
    pi = np.pi;
    ncontours = 100
    eph = np.linspace(9831.65, 9832.25, 5001);
    theta_0 = 45.0 * pi / 180.;
    dtheta_0 = 50e-6;
    ntheta_0 = 101;
    # theta_0 *= 0.
    theta = np.linspace(theta_0 - dtheta_0, theta_0 + dtheta_0, ntheta_0);

    Eph, Thetaurad = np.meshgrid(eph, 1e6 * (theta - pi / 4));

    R0H, R00 = Bragg_mirror(photon_energies_eV=eph, angles_rad=theta, d=d)

    plt.contourf(Eph, Thetaurad, np.abs(R0H) ** 2, ncontours)
    plt.ylabel('Angle - 45 deg (urad)')
    plt.xlabel('Photon energy (eV)')
    plt.title('Bragg diffration intensity')
    plt.colorbar();
    plt.show()

    plt.contourf(Eph, Thetaurad, np.abs(R00) ** 2, ncontours)
    plt.ylabel('Angle - 45 deg (urad)')
    plt.xlabel('Photon energy (eV)')
    plt.title('Forward diffration intensity')
    plt.colorbar();
    plt.show()

    # plt.contourf(Eph,Thetaurad,np.abs(C)**2,ncontours)
    # plt.ylabel('Angle - 45 deg (urad)')
    # plt.xlabel('Photon energy (eV)')
    # plt.title('intensity(prompt response)')
    # plt.colorbar(); plt.show()

    # plt.contourf(Eph,Thetaurad,np.abs(R001)**2,ncontours)
    # plt.ylabel('Angle - 45 deg (urad)')
    # plt.xlabel('Photon energy (eV)')
    # plt.title('intensity(forward diffration - prompt response)')
    # plt.colorbar(); plt.show()


def get_Bragg_mirror_response_eph(eph, theta_0=np.pi / 4., d=100e-6):
    R0H, R00 = Bragg_mirror(photon_energies_eV=eph, angles_rad=theta_0, d=d)
    print('test')
    return R0H.squeeze(), R00.squeeze()


def get_Bragg_mirror_response_theta(theta, eph_0=9831.2, d=100e-6):

    R0H, R00 = Bragg_mirror(photon_energies_eV=eph_0, angles_rad=theta, d=d)

    return R0H.squeeze(), R00.squeeze()