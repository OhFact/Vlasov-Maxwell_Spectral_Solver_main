import jax.numpy as jnp
import jax
import time
import sys, os
from jax.scipy.optimize import minimize
from scipy.signal import find_peaks
from jax.numpy.fft import fft, ifftn, fftshift, ifftshift, fftfreq
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from JAX_Vlasov_Maxwell_solver import *
jax.config.update("jax_enable_x64", True)

Nx, Ny, Nz = 3, 1, 1
Nvx, Nvy, Nvz = 10, 10, 10
Lx, Ly, Lz = 1.0, 1.0, 1.0
Nn, Nm, Np, Ns = 80, 1, 1, 2
nu, omega_ce = 1.0, 1.0
mi_me, Ti_Te = 1000000.0, 10
Omega_cs =  omega_ce * jnp.array([1.0, 1.0 / mi_me])
alpha_e = [0.1, 0.1, 0.1]
alpha_s = jnp.concatenate([jnp.array(alpha_e), (jnp.array(alpha_e) * jnp.sqrt(Ti_Te / mi_me))])
qs = jnp.array([-1, 1])
u_s = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
t_steps, t_max = 2001, 200
dt = int((t_steps-1)/t_max)

def Landau_damping_HF_1D(Lx, Ly, Lz, Omega_ce, alpha_e, alpha_i, Nn):
    """
    I have to add docstrings!
    """

    vte = alpha_e / jnp.sqrt(2)  # Electron thermal velocity.
    vti = alpha_i / jnp.sqrt(2)  # Ion thermal velocity.

    kx = 2 * jnp.pi / Lx  # Wavenumber.

    dn = 0.0001  # Density fluctuation.

    # Fourier components of magnetic and electric fields.
    Fk_0 = jnp.zeros((6, 1, 3, 1), dtype=jnp.complex128)
    Fk_0 = Fk_0.at[0, 0, 0, 0].set(dn / (2 * kx))
    Fk_0 = Fk_0.at[0, 0, 2, 0].set(dn / (2 * kx))
    #    Fk_0 = Fk_0.at[3, 0, 1, 0].set(Omega_ce)

    # Hermite-Fourier components of electron and ion distribution functions.
    Ce0_mk, Ce0_0, Ce0_k = 0 + 1j * (1 / 2 ** (5 / 2)) * (1 / vte ** 3) * dn, 1 / (
                (2 ** (3 / 2)) * (vte ** 3)) + 0 * 1j, 0 - 1j * (1 / 2 ** (5 / 2)) * (1 / vte ** 3) * dn
    Ci0_0 = 1 / ((2 ** (3 / 2)) * (vti ** 3)) + 0 * 1j
    Ck_0 = jnp.zeros((2 * Nn, 1, 3, 1), dtype=jnp.complex128)
    Ck_0 = Ck_0.at[0, 0, 0, 0].set(Ce0_mk)
    Ck_0 = Ck_0.at[0, 0, 1, 0].set(Ce0_0)
    Ck_0 = Ck_0.at[0, 0, 2, 0].set(Ce0_k)
    Ck_0 = Ck_0.at[Nn, 0, 1, 0].set(Ci0_0)

    return Ck_0, Fk_0

#Timing of the simulation
start_time = time.time()
Ck_0, Fk_0 = Landau_damping_HF_1D(Lx, Ly, Lz, Omega_cs[0], alpha_s[0], alpha_s[3], Nn)
Ck, Fk, t = VM_simulation(Ck_0, Fk_0, qs, nu, Omega_cs, alpha_s, mi_me, u_s, Lx, Ly, Lz, Nx, Ny, Nz, Nn, Nm, Np, Ns, t_max, t_steps)
end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")

"""
1D Landau damping/two-stream stability: data analysis.
"""

#Compute Plasma Vars
lambda_D = jnp.sqrt(1 / (2 * (1 / alpha_s[0] ** 2 + 1 / (mi_me * alpha_s[3] ** 2))))
k_norm = jnp.sqrt(2) * jnp.pi * alpha_s[0] / Lx

#Set boundary conditions
dCk = Ck.at[:, 0, 1, 0, 0].set(0)
dCk = dCk.at[:, Nn * Nm * Np, 1, 0, 0].set(0)

# Save mean energy distribution over vel space
C2 = jnp.mean(jnp.abs(dCk) ** 2, axis={-3, -2, -1})

plasma_energy_0_Ck = (0.5 * ((0.5 * (alpha_s[0] ** 2 + alpha_s[1] ** 2 + alpha_s[2] ** 2)) *
                                        alpha_s[0] * alpha_s[1] * alpha_s[2] * Ck[:, 0, 1, 0, 0].real) +
                                0.5 * mi_me * ((0.5 * (alpha_s[3] ** 2 + alpha_s[4] ** 2 + alpha_s[5] ** 2)) *
                                                alpha_s[3] * alpha_s[4] * alpha_s[5] * Ck[:, Nn, 1, 0, 0].real))

plasma_energy_2_Ck = (0.5 * (1 / jnp.sqrt(2)) * (alpha_s[0] ** 2) * Ck[:, 2, 1, 0, 0].real * alpha_s[0] * alpha_s[1] * alpha_s[2] +
                                0.5 * mi_me * (1 / jnp.sqrt(2)) * (alpha_s[3] ** 2) * Ck[:, Nn + 2, 1, 0, 0].real * alpha_s[3] * alpha_s[4] * alpha_s[5])

#Electric field energy density
electric_energy_Fk = 0.5 * jnp.mean(Fk[:, 0, :, 0, 0] ** 2, axis=-1) * Omega_cs[0] ** 2

"""
Function to Fit
"""

# Define the function to fit
def model_function(params, t):
    A, B, omega, gamma = params
    return A * jnp.cos(omega * t) * jnp.exp(-gamma * t) + B

# Define the loss function (mean squared error)
def loss_function(params, t, data):
    predictions = model_function(params, t)
    return jnp.mean((predictions - data) ** 2)

# Initialize parameters (A, B, omega, gamma) with some reasonable guesses
initial_params = jnp.array([5.0, 0.0, 1.0, 0.2])

## Example time array and data array (replace 'data' with your actual data array)

# Minimize the loss function to find the best-fit parameters, method replaceable
optimized_result = minimize(lambda params: loss_function(params, t, dCk[:, 0, 0, 0, 0].imag), initial_params, method='BFGS', tol=1e-9)

# Extract the best-fit parameters
best_fit_params = optimized_result.x
A, B, omega, gamma = best_fit_params

# Compute Fast fourier transform of the perturbation signals
dCek_freq = fftshift(fft(dCk[:, 0, 0, 0, 0].imag))
cos_exp_freq = fftshift(fft(jnp.cos(omega * t) * jnp.exp(-gamma * t)))
freq = fftshift(fftfreq(len(dCk[:, 0, 0, 0, 0].imag), 0.1))
max_index = jnp.argmax(dCek_freq.real)

# Find peaks of signals
peaks, _ = find_peaks(jnp.abs(dCk[:, 0, 0, 0, 0].imag))
p = jnp.polyfit(t[peaks], jnp.log(jnp.abs(dCk[:, 0, 0, 0, 0].imag[peaks])), 1)

"""

Edit Plots below, choose which ones to run and which not to

"""
plot_C_vs_t(t, dCk, Nn, Nm, Np, k_norm, u_s, lambda_D, mi_me, 10)
plot_Ci_vs_t(t, dCk, Nn, Nm, Np, k_norm, u_s, lambda_D, mi_me, 10)
plot_rho_k_vs_t(t, dCk, Nn, Nm, Np, alpha_s, k_norm, u_s, lambda_D, mi_me)
#plot_C000_frequency_spectrum(freq, dCek_freq, nu, k_norm, lambda_D, mi_me, Nn, cos_exp_freq)
plot_C2_vs_n_vs_t(C2, Nn, Nm, Np, nu, Lx, lambda_D, mi_me)
plot_energy(t, electric_energy_Fk, plasma_energy_2_Ck, nu, Lx, lambda_D, mi_me, Nn)