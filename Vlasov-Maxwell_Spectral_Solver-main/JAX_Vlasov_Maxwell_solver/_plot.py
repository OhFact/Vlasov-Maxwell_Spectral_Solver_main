import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_C_vs_t(t, dCk, Nn, Nm, Np, k_norm, u_s, lambda_D, mi_me, mode):
    """
    Plot |C| vs t.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, jnp.log10(jnp.abs(dCk[:, mode, 0, 0, 0].imag)), label='$log_{10}(|\delta C_{e000,k}|)$', linestyle='-', color='red', linewidth=3.0)
    plt.ylabel(r'$log_{10}(|\delta C_{e000,k}|)$', fontsize=16)
    plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
    plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.show()

def plot_Ci_vs_t(t, dCk, Nn, Nm, Np, k_norm, u_s, lambda_D, mi_me, mode):
    """
    Plot |Ci| vs t.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, jnp.log10(jnp.abs(dCk[:, Nn * Nm * Np + mode, 0, 0, 0].imag)), label='$log_{10}(|\delta C_{i000,k}|)$', linestyle='-', color='red', linewidth=3.0)
    plt.ylabel(r'$log_{10}(|\delta C_{i000,k}|)$', fontsize=16)
    plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
    plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.show()

def plot_rho_k_vs_t(t, dCk, Nn, Nm, Np, alpha_s, k_norm, u_s, lambda_D, mi_me):
    """
    Plot |rho_k| vs t.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, jnp.log10(jnp.abs(dCk[:, Nn * Nm * Np, 0, 0, 0].imag * (alpha_s[3] * alpha_s[4] * alpha_s[5]) -
                                     dCk[:, 0, 0, 0, 0].imag * (alpha_s[0] * alpha_s[1] * alpha_s[2]))),
         label='$log_{10}(|\rho_k|)$', linestyle='-', color='red', linewidth=3.0)
    plt.ylabel(r'$log_{10}(|\rho_k|)$', fontsize=16)
    plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
    plt.title(rf'$kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, u_e/c = {u_s[0]}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.show()

def plot_C000_frequency_spectrum(freq, dCek_freq, nu, k_norm, lambda_D, mi_me, Nn, cos_exp_freq):
    """
    Plot C000 frequency spectrum.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(2 * jnp.pi * freq, dCek_freq.imag, label='$Im[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='red', linewidth=3.0)
    plt.ylabel(r'$\hat{\delta C}_{e000,\omega}$', fontsize=16)
    plt.xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
    plt.title(rf'$\nu ={nu}, kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.legend().set_draggable(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(2 * jnp.pi * freq, cos_exp_freq.real, label='$Im[\delta C_{e000,\omega}]$', linestyle='None', marker='x', color='red', linewidth=3.0)
    plt.ylabel(r'$Real[\hat{\delta C}_{e000,\omega}]$', fontsize=16)
    plt.xlabel(r'$\omega/\omega_{pe}$', fontsize=16)
    plt.title(rf'$\nu ={nu}, kv_{{th,e}}/\omega_{{pe}} = {k_norm:.2}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.show()

def plot_C2_vs_n_vs_t(C2, Nn, Nm, Np, nu, Lx, lambda_D, mi_me):
    """
    Plot |C|^2 vs n vs t.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(jnp.log10(C2[:, :Nn * Nm * Np]), aspect='auto', cmap='viridis',
               interpolation='none', origin='lower', extent=(0, Nn, 0, 100), vmin=-10, vmax=0)
    plt.colorbar(label=r'$log_{10}(\langle |C_{e,n}|^2\rangle (t))$').ax.yaxis.label.set_size(16)
    plt.xlabel('n', fontsize=16)
    plt.ylabel('t', fontsize=16)
    plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=14)
    plt.show()

def plot_energy(t, electric_energy_Fk, plasma_energy_2_Ck, nu, Lx, lambda_D, mi_me, Nn):
    """
    Plot energy.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t[:], electric_energy_Fk[:] + plasma_energy_2_Ck[:] / 3, label='Total energy in fluctuations', linestyle='-', color='red', linewidth=3.0)
    plt.xlabel(r'$t\omega_{pe}$', fontsize=16)
    plt.ylabel(r'Energy', fontsize=16)
    plt.xlim((0, t[-1]))
    plt.title(rf'$\nu ={nu}, L_x/d_e = {Lx}, \lambda_D/d_e = {lambda_D:.1e}, m_i/m_e = {mi_me}, N_n = {Nn}$', fontsize=16)
    plt.legend().set_draggable(True)
    plt.show()

def plot_Kelvin_Helmholtz_instability(We, Lx, Ly):
    """
    Plot Kelvin-Helmholtz instability.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(We[0, ...], aspect='auto', cmap='viridis',
               interpolation='none', origin='lower', extent=(0, Lx, 0, Ly))
    plt.colorbar(label=r'$U_{ex}$').ax.yaxis.label.set_size(16)
    plt.xlabel('x/d_e', fontsize=16)
    plt.ylabel('y/d_e', fontsize=16)
    plt.show()

def animate_Kelvin_Helmholtz_instability(We):
    """
    Animate Kelvin-Helmholtz instability.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(We[0], cmap='viridis', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("$U_{ex}/c$")
    title = ax.set_title("Frame 0")
    ax.axis('off')

    def update(frame):
        im.set_array(We[frame])
        title.set_text(f"Frame {frame}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=We.shape[0], interval=50, blit=True)
    plt.show()

def plot_plasma_energy(t, plasma_energy):
    """
    Plot plasma energy.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(t, plasma_energy, label='plasma energy', linestyle='-', color='red', linewidth=3.0)
    plt.show()