import numpy as np
# import numpy.fft as npfft
# import scipy as sp
from muFFT import FFT
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib as mpl
# from matplotlib import cbook, cm
# import matplotlib.colors as mpl_colors
from scipy.stats import binned_statistic

mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['figure.figsize'] = [8, 8]

t = 0
M = 5
N = 2**M
L = 2*np.pi
X = np.mgrid[:N, :N, :N].astype(float)*L/N

nb_grid_pts = (N, N, N)
nx, ny, nz = nb_grid_pts
lx, ly, lz = L, L, L
fft = FFT(nb_grid_pts, engine="mpi", allow_temporary_buffer=True, communicator=MPI.COMM_WORLD)

K = (2 * np.pi * fft.ifftfreq.T / np.array([lx, ly, lz])).T
K2 = np.square(np.linalg.norm(K, axis=0))
KoverK2 = np.divide(K, np.where(K2 == 0, 1.0, K2))

# Assuming we manually set the velocity amplitude
velocity_amplitude = 0.001

# Initialize random velocity field
wavevector = (2 * np.pi * fft.fftfreq.T / np.array([lx, ly, lz])).T
zero_wavevector = (wavevector.T == np.zeros(3, dtype=int)).T.all(axis=0)
wavevector_sq = np.sum(wavevector ** 2, axis=0)
# Fourier space velocity field
random_field = np.zeros((3,) + fft.nb_fourier_grid_pts, dtype=complex)
rng = np.random.default_rng()
random_field.real = rng.standard_normal(random_field.shape)
random_field.imag = rng.standard_normal(random_field.shape)
fac = np.zeros_like(wavevector_sq)
# Avoid division by zero
fac[np.logical_not(zero_wavevector)] = velocity_amplitude * wavevector_sq[np.logical_not(zero_wavevector)] ** (-5 / 6)
random_field *= fac

# Dealiasing following Mikael Mortensen and Hans Petter Langtangen
kmax_dealias = 2.0 / 3.0 * (N/2+1)
dealias = np.array((abs(K[0]) < kmax_dealias) * (abs(K[1]) < kmax_dealias) * (abs(K[2]) < kmax_dealias), dtype=bool)

k_low_threshold = 2
low_wavenumber_mask = np.square(np.linalg.norm(K, axis=0)) < k_low_threshold

# print(random_field.shape)
# print(random_field)

random_f_field = fft.fourier_space_field('U_ff', (3,))
random_r_field = fft.real_space_field('U_rf', (3,))

random_f_field.p = random_field.copy()

fft.ifft(random_f_field, random_r_field)

# print(random_r_field.p.shape)
# print(random_r_field.p)

freeze = True

# input()

U = np.zeros((3, N, N, N))
# U[0] = np.cos(X[0]) * np.sin(X[1])
# U[1] = -np.sin(X[0]) * np.cos(X[1])

U = random_r_field.p.copy()


# Analytical solution to Taylorgreen vortex; y is the initial condition
def analytical_solution(t, nu, y):
    """
    Computes the analytical solution to the Taylor Green vortex problem.

    Parameters
    ----------
    t : float
        The time at which to evaluate the solution.
    nu : float
         The kinematic viscosity of the fluid.
    y : array_like
        The initial condition for the velocity field.
    
    Returns
    -------
    yt : np.ndarray
         The velocity field at time t.
    """
    return y * np.exp(-2 * nu * t)


def rk4(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Implements the fourth-order Runge-Kutta method for numerical integration
    of multidimensional fields.

    Parameters
    ----------
    f : function
        The function to be integrated. It should take two arguments: time t
        and field y.
    t : float
        The current time.
    y : array_like
        The current value of the field.
    dt : float
        The time step for the integration.

    Returns
    -------
    dy : np.ndarray
        The increment of the field required to obtain the value at t + dt.
    """
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return dt * (1/6) * (k1 + (2 * k2) + (2 * k3) + k4)


def Curl3D(u_f_field_p, curl_f_field_p, curl_r_field_p):
    """
    Computes the curl of a 3D vector field in Fourier space and performs an inverse Fourier transform.

    Parameters
    ----------
    u_f_field_p : array_like
                  The input 3D velocity field in Fourier space.
    curl_f_field_p : array_like
                     Placeholder for the curl of the input field in Fourier space.
                     Gets modified in-place.
    curl_r_field_p : array_like
                     Placeholder for the curl of the input field in real space.
                     Gets modified in-place.

    Returns
    -------
    None.
    """
    u_x = u_f_field_p[0]
    u_y = u_f_field_p[1]
    u_z = u_f_field_p[2]

    k_x = fft.fftfreq[0]
    k_y = fft.fftfreq[1]
    k_z = fft.fftfreq[2]

    curl_f_field_p[0] = 1j * (k_y * u_z - k_z * u_y)
    curl_f_field_p[1] = 1j * (k_z * u_x - k_x * u_z)
    curl_f_field_p[2] = 1j * (k_x * u_y - k_y * u_x)

    fft.ifft(curl_f_field_p, curl_r_field_p)


def Cross3D(a, b, result_r_field_p, result_f_field_p):
    """
    Computes the cross product of two 3D vectors in real space and performs a Fourier transform.

    Parameters
    ----------
    a : array_like
        The first 3D vector.
    b : array_like
        The second 3D vector.
    result_r_field_p : array_like
                          Placeholder for the result of the cross product in real space.
                          Gets modified in-place.
    result_f_field_p : array_like
                            Placeholder for the result of the cross product in Fourier space.
                            Gets modified in-place.

    Returns
    -------
    None.
    """
    result_r_field_p[0] = a[1]*b[2] - a[2]*b[1]
    result_r_field_p[1] = a[2]*b[0] - a[0]*b[2]
    result_r_field_p[2] = a[0]*b[1] - a[1]*b[0]

    fft.fft(result_r_field_p, result_f_field_p)

    result_f_field_p[0] *= fft.normalisation
    result_f_field_p[1] *= fft.normalisation
    result_f_field_p[2] *= fft.normalisation


def computeRHS(t, U):
    """
    Computes the right-hand side of the Navier-Stokes equations partially in Fourier space.
    At the end, the result is transformed back to real space.

    Parameters
    ----------
    t : float
        The current time.
    U : array_like
        The current velocity field in real space.

    Returns
    -------
    result_r_field.p : np.ndarray
        The right-hand side of the Navier-Stokes equations in real space.
    """

    # U_r_field.p = U
    U_r_field.p[0] = U[0]
    U_r_field.p[1] = U[1]
    U_r_field.p[2] = U[2]

    fft.fft(U_r_field, U_f_field)

    # Normalise the Fourier transform
    U_f_field.p[0] *= fft.normalisation
    U_f_field.p[1] *= fft.normalisation
    U_f_field.p[2] *= fft.normalisation
    # U_f_field.p is now the velocity field in Fourier space

    Curl3D(U_f_field.p, Curl_f_field.p, Curl_r_field.p)
    # Curl_f_field.p is now the curl of the velocity field in Fourier space
    # Curl_r_field.p is now the curl of the velocity field in real space
    Cross3D(U, Curl_r_field.p, cross_result_r_field.p, cross_result_f_field.p)
    # cross_result_r_field.p is now the cross product of U and the curl of U in real space

    # Dealiasing following Mikael Mortensen and Hans Petter Langtangen
    cross_result_f_field.p *= dealias

    # Follows definition of the pressure term in Fourier space
    Pressure_f_field.p = cross_result_f_field.p.copy()*KoverK2

    cross_result_f_field.p[0] -= Pressure_f_field.p[0]*K[0]
    cross_result_f_field.p[1] -= Pressure_f_field.p[1]*K[1]
    cross_result_f_field.p[2] -= Pressure_f_field.p[2]*K[2]

    cross_result_f_field.p[0] -= nu*K2*U_f_field.p[0]
    cross_result_f_field.p[1] -= nu*K2*U_f_field.p[1]
    cross_result_f_field.p[2] -= nu*K2*U_f_field.p[2]

    if freeze:
        cross_result_f_field.p[:, low_wavenumber_mask] = 0

    fft.ifft(cross_result_f_field, result_r_field)

    return result_r_field.p.copy()


# Same as computeRHS but without dealiasing
def computeRHSnoDealias(t, U):
    """
    Computes the right-hand side of the Navier-Stokes equations partially in Fourier space.
    At the end, the result is transformed back to real space.
    Same as computeRHS but without dealiasing.

    Parameters
    ----------
    t : float
        The current time.
    U : array_like
        The current velocity field in real space.

    Returns
    -------
    result_r_field.p : np.ndarray
        The right-hand side of the Navier-Stokes equations in real space.
    """

    U_r_field.p[0] = U[0]
    U_r_field.p[1] = U[1]
    U_r_field.p[2] = U[2]

    fft.fft(U_r_field, U_f_field)

    U_f_field.p[0] *= fft.normalisation
    U_f_field.p[1] *= fft.normalisation
    U_f_field.p[2] *= fft.normalisation

    Curl3D(U_f_field.p, Curl_f_field.p, Curl_r_field.p)
    Cross3D(U, Curl_r_field.p, cross_result_r_field.p, cross_result_f_field.p)

    Pressure_f_field.p = cross_result_f_field.p.copy()*KoverK2

    cross_result_f_field.p[0] -= Pressure_f_field.p[0]*K[0]
    cross_result_f_field.p[1] -= Pressure_f_field.p[1]*K[1]
    cross_result_f_field.p[2] -= Pressure_f_field.p[2]*K[2]

    cross_result_f_field.p[0] -= nu*K2*U_f_field.p[0]
    cross_result_f_field.p[1] -= nu*K2*U_f_field.p[1]
    cross_result_f_field.p[2] -= nu*K2*U_f_field.p[2]

    if freeze:
        cross_result_f_field.p[:, low_wavenumber_mask] = 0

    fft.ifft(cross_result_f_field, result_r_field)

    return result_r_field.p.copy()


def energy_spectrum(u_):
    E_k = 0.5 * (np.linalg.norm(u_, axis=0)**2)
    E_k[zero_wavevector] = 0  # Ignore zero mode
    return E_k


def dissipation_spectrum(u_):
    D_k = nu * K2 * (np.linalg.norm(u_, axis=0)**2)
    D_k[zero_wavevector] = 0  # Ignore zero mode
    return D_k


# Pre initialize all fields for FFT
U_r_field = fft.real_space_field('U_rf', (3,))
U_f_field = fft.fourier_space_field('U_ff', (3,))
cross_result_r_field = fft.real_space_field('Cross_result_rf', (3,))
cross_result_f_field = fft.fourier_space_field('Cross_result_ff', (3,))
result_r_field = fft.real_space_field('Result_rf', (3,))
Pressure_f_field = fft.fourier_space_field('Pressure_ff', 3)
Curl_r_field = fft.real_space_field('Curl_rf', (3,))
Curl_f_field = fft.fourier_space_field('Curl_ff', (3,))

# Parameters
nu = (1/1600)
maxT = 40
td = 0.01
maxSteps = int(maxT/td)+1

# Counting variables
t = 0
step = 0

# Get baseline for analytical solution (this is/was for Taylorgreen Vortex)
U_old = U.copy()
dU_ana = U_old - analytical_solution(t, nu, U_old)

# Get baseline for undealiased solution
U_aliased = U.copy()

for i in range(maxSteps):

    dU = rk4(computeRHS, t, U, td)
    dU_aliased = rk4(computeRHSnoDealias, t, U_aliased, td)

    if i == 0:
        dU_ana = U_old - analytical_solution(t, nu, U_old)
    else: 
        dU_ana = U_ana - analytical_solution(t, nu, U_old)

    U_ana = analytical_solution(t, nu, U_old)

    du_error = np.linalg.norm(dU - dU_aliased)
    # print("Error in du: ", du_error)

    U += dU
    U_aliased += dU_aliased

    if step % 100 == 0:
        # U_ana = analytical_solution(t, nu, U_old)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Numerical Solution")
        plt.quiver(X[0][:, :, N//2], X[1][:, :, N//2], U_aliased[0][:, :, N//2], U_aliased[1][:, :, N//2], scale=30)

        plt.subplot(1, 2, 2)
        plt.title("Numerical Solution (with dealiasing)")
        plt.quiver(X[0][:, :, N//2], X[1][:, :, N//2], U[0][:, :, N//2], U[1][:, :, N//2], scale=30)

        # error = np.linalg.norm(U - U_ana)
        # plt.suptitle(f"Time: {t}, Error: {error}")
        plt.savefig(f'velocity_field_comp_{step}.png')
        plt.close()

    # Save velocity field to file every 100 steps
    # if step % 100 == 0:
        print(f"Step: {step}, Time: {t}")
        np.save(f'velocity_field_{step}.npy', U)

        U_f = fft.fft(U) * fft.normalisation
        U_f_aliased = fft.fft(U_aliased) * fft.normalisation

        # Calculate energy and dissipation spectra
        E_k = energy_spectrum(U_f)
        D_k = dissipation_spectrum(U_f)

        E_k_aliased = energy_spectrum(U_f_aliased)
        D_k_aliased = dissipation_spectrum(U_f_aliased)

        # print(E_k.shape)
        # print(D_k.shape)

        k_n_sq = np.linalg.norm(K, axis=0)**2

        x_min = max(np.min(k_n_sq), 1/N)
        x_max = np.max(k_n_sq)
        x_space = np.linspace(x_min, x_max, N)

        # print(k_n_sq.flatten().shape)
        # print(E_k.flatten().shape)

        k_n_sq_flat = k_n_sq.flatten()
        E_k_flat = E_k.flatten()
        D_k_flat = D_k.flatten()
        E_k_aliased_flat = E_k_aliased.flatten()
        D_k_aliased_flat = D_k_aliased.flatten()

        bins = np.logspace(np.log10(x_min), np.log10(x_max), 50)

        bin_means, bin_edges, binnumber = binned_statistic(x=k_n_sq_flat, values=E_k_flat, bins=bins)
        dis_bin_means, dis_bin_edges, dis_binnumber = binned_statistic(x=k_n_sq_flat, values=D_k_flat, bins=bins)

        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2

        dis_bin_width = (dis_bin_edges[1] - dis_bin_edges[0])
        dis_bin_centers = dis_bin_edges[1:] - dis_bin_width/2

        aliased_bin_means, aliased_bin_edges, aliased_binnumber = binned_statistic(x=k_n_sq_flat,
                                                                                   values=E_k_aliased_flat,
                                                                                   bins=bins)
        aliased_dis_bin_means, aliased_dis_bin_edges, aliased_dis_binnumber = binned_statistic(x=k_n_sq_flat,
                                                                                               values=D_k_aliased_flat,
                                                                                               bins=bins)

        aliased_bin_width = (aliased_bin_edges[1] - aliased_bin_edges[0])
        aliased_bin_centers = aliased_bin_edges[1:] - aliased_bin_width/2

        aliased_dis_bin_width = (aliased_dis_bin_edges[1] - aliased_dis_bin_edges[0])
        aliased_dis_bin_centers = aliased_dis_bin_edges[1:] - aliased_dis_bin_width/2

        # Way too many plots, checked which type seems to fit best in the report.

        plt.figure()
        plt.loglog(bin_centers, bin_means,  "g", linestyle='dashed')
        plt.loglog(dis_bin_centers, dis_bin_means, "b", linestyle='dashdot')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')
        plt.legend(["E(q)", "D(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Energy Spectrum E(q) ($cm^{3}$/$s^{2}$), Dissipation Spectrum D(q) ($cm^{2}$/$s^{3}$) ")
        plt.xlim([1, None])
        plt.savefig(f'spectra_{step}.png')
        plt.close()

        plt.figure()
        plt.loglog(bin_centers, bin_means,  "g", linestyle='dashed')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')
        plt.legend(["E(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Energy Spectrum E(q) ($cm^{3}$/$s^{2}$)")
        plt.xlim([1, None])
        plt.savefig(f'energy_spectrum_{step}.png')
        plt.figure()
        plt.loglog(dis_bin_centers, dis_bin_means,  "b", linestyle='dashed')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')

        plt.legend(["D(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Dissipation Spectrum D(q) ($cm^{2}$/$s^{3}$)")
        plt.xlim([1, None])
        plt.savefig(f'dissipate_spectrum_{step}.png')
        plt.close()

        plt.figure()
        plt.loglog(aliased_bin_centers, aliased_bin_means,  "g", linestyle='dashed')
        plt.loglog(aliased_dis_bin_centers, aliased_dis_bin_means, "b", linestyle='dashdot')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')
        plt.legend(["E(q)", "D(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Energy Spectrum E(q) ($cm^{3}$/$s^{2}$), Dissipation Spectrum D(q) ($cm^{2}$/$s^{3}$)")
        plt.xlim([1, None])
        plt.savefig(f'aliased_spectra_{step}.png')
        plt.close()

        plt.figure()
        plt.loglog(aliased_bin_centers, aliased_bin_means,  "g", linestyle='dashed')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')
        plt.legend(["E(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Energy Spectrum E(q) ($cm^{3}$/$s^{2}$)")
        plt.xlim([1, None])
        plt.savefig(f'aliased_energy_spectrum_{step}.png')
        plt.close()

        plt.figure()
        plt.loglog(aliased_dis_bin_centers, aliased_dis_bin_means,  "b", linestyle='dashed')
        plt.plot(x_space, x_space**(-5/3), "r", linestyle='solid')
        plt.legend(["D(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Dissipation Spectrum D(q) ($cm^{2}$/$s^{3}$)")
        plt.xlim([1, None])
        plt.savefig(f'aliased_dissipate_spectrum_{step}.png')
        plt.close()

        plt.figure()
        plt.loglog(bin_centers, bin_means,  "r", linestyle='dashed')
        plt.loglog(aliased_dis_bin_centers, aliased_dis_bin_means, "g", linestyle=(0, (3, 1, 1, 1, 1, 1)))
        plt.loglog(aliased_bin_centers, aliased_bin_means,  "b", linestyle='dashdot')
        plt.loglog(dis_bin_centers, dis_bin_means, "y", linestyle='dotted')
        plt.plot(x_space, x_space**(-5/3), "k", linestyle='solid')
        plt.legend(["E(q) (dealiased)", "D(q) (dealiased)", "E(q)", "D(q)", r"$q^{\frac{-5}{3}}$"])
        plt.xlabel("Wavenumber q ($cm^{-1}$)")
        plt.ylabel(r"Energy Spectra E(q) ($cm^{3}$/$s^{2}$), Dissipation Spectra D(q) ($cm^{2}$/$s^{3}$)")
        plt.xlim([1, None])
        plt.savefig(f'all_spectra_{step}.png')
        plt.close()

        # Visualization every 500 steps
    # if step % 100 == 0:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Velocity Field Slice X-Y Plane")
        plt.quiver(X[0][:, :, N//2], X[1][:, :, N//2], U[0][:, :, N//2], U[1][:, :, N//2], scale=30)

        plt.subplot(1, 2, 2)
        plt.title("Velocity Field Slice Y-Z Plane")
        plt.quiver(X[1][N//2, :, :], X[2][N//2, :, :], U[1][N//2, :, :], U[2][N//2, :, :], scale=30)

        plt.suptitle(f"Time: {t}")
        plt.savefig(f'velocity_field_{step}.png')
        plt.close()

        c = np.sqrt(np.abs(U[0]) ** 2 + np.abs(U[1]) ** 2 + np.abs(U[2]) ** 2)
        c = (c.ravel() - c.min()) / np.ptp(c)
        # Repeat for each body line and two head lines
        c = np.concatenate((c, np.repeat(c, 2)))
        # Colormap
        c = plt.cm.jet(c)

        ax = plt.figure(figsize=(32, 32)).add_subplot(projection='3d')
        ax.quiver(X[0], X[1], X[2], U[0], U[1], U[2], colors=c, normalize=True, pivot='middle', length=0.1)
        plt.savefig(f'velocity_field_3d_{step}.png')
        plt.close()
        # plt.show()

    step += 1
    t += td
