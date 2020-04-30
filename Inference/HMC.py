import torch

import numpy as np
import scipy.stats as st
from tqdm import tqdm, trange

from numpy.linalg import norm


"""
adapted from
https://github.com/ColCarroll/minimc.git
"""

def leapfrog(q, p, dVdq, potential, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.
    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : np.floatX
        Gradient of the potential at the initial coordinates
    potential : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    p -= step_size * dVdq / 2  # half step
    for _ in np.arange(path_len): #np.arange(np.round(path_len / step_size) - 1):
        q += step_size * p  # whole step
        V, dVdq = potential(q)
        p -= step_size * dVdq  # whole step
    q += step_size * p  # whole step
    V, dVdq = potential(q)
    p -= step_size * dVdq / 2  # half step

    # momentum flip at end
    return q, -p, V, dVdq


def leapfrog_twostage(q, p, dVdq, potential, path_len, step_size):
    """A second order symplectic integration scheme.
    Based on the implementation from Adrian Seyboldt in PyMC3. See
    https://github.com/pymc-devs/pymc3/pull/1758 for a discussion.
    References
    ----------
    Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
    Integrators for the Hybrid Monte Carlo Method." SIAM Journal on
    Scientific Computing 36, no. 4 (January 2014): A1556-80.
    doi:10.1137/130932740.
    Mannseth, Janne, Tore Selland Kleppe, and Hans J. Skaug. "On the
    Application of Higher Order Symplectic Integrators in
    Hamiltonian Monte Carlo." arXiv:1608.07048 [Stat],
    August 25, 2016. http://arxiv.org/abs/1608.07048.
    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : np.floatX
        Gradient of the potential at the initial coordinates
    potential : callable
        Value and gradient of the potential
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be
    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    a = (3 - np.sqrt(3)) / 6

    p -= a * step_size * dVdq  # `a` momentum update
    for _ in np.arange(path_len): #np.arange(np.round(path_len / step_size) - 1):
        q += step_size * p / 2  # half position update
        V, dVdq = potential(q)
        p -= (1 - 2 * a) * step_size * dVdq  # 1 - 2a position update
        q += step_size * p / 2  # half position update
        V, dVdq = potential(q)
        p -= 2 * a * step_size * dVdq  # `2a` momentum update
    q += step_size * p / 2  # half position update
    V, dVdq = potential(q)
    p -= (1 - 2 * a) * step_size * dVdq  # 1 - 2a position update
    q += step_size * p / 2  # half position update
    V, dVdq = potential(q)
    p -= a * step_size * dVdq  # `a` momentum update

    return q, -p, V, dVdq


def hamiltonian_monte_carlo(
    numiter,
    burnin,
    thinning,
    potential,
    initial_position,
    check_rate=500,
    initial_potential=None,
    initial_potential_grad=None,
    path_len=1,
    initial_step_size=0.1,
    integrator=leapfrog,
    max_energy_change=1000.0,
):
    """Run Hamiltonian Monte Carlo sampling.
    Parameters
    ----------
    n_samples : int
        Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    tune: int
        Number of iterations to run tuning
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    initial_step_size : float
        How long each integration step is. This will be tuned automatically.
    max_energy_change : float
        The largest tolerable integration error. Transitions with energy changes
        larger than this will be declared divergences.
    Returns
    -------
    np.array
        Array of length `n_samples`.
    """
    acceptance_count=[] # record number of accept for the last check_rate iterations
    current_acceptance_rate=0.
    initial_position = np.array(initial_position)
    if initial_potential is None or initial_potential_grad is None:
        initial_potential, initial_potential_grad = potential(initial_position)

    q_last=initial_position
    # collect all our samples in a list
    samples = []
    accept_rates= []
    step_sizes=[]
    log_prob=[]
    
    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    step_size = initial_step_size
    
    with trange(numiter) as tr:
        for t in tr:
            p0=momentum.rvs(size=initial_position.shape[:1])
            # Integrate over our path to get a new position and momentum
            q_new, p_new, final_V, final_dVdq = integrator(
                q_last,
                p0,
                initial_potential_grad,
                potential,
                path_len=2* np.random.rand()* path_len,  # We jitter the path length a bit
                step_size=step_size,
            )

            start_log_p = np.sum(momentum.logpdf(p0)) - initial_potential
            new_log_p = np.sum(momentum.logpdf(p_new)) - final_V
            energy_change = new_log_p - start_log_p

            # Check Metropolis acceptance criterion
            p_accept = min(1, np.exp(energy_change))
            if np.random.rand() < p_accept:
                acceptance_count.append(1.)
                initial_potential = final_V
                initial_potential_grad = final_dVdq
            else:
                q_new=q_last
                acceptance_count.append(0.)

            if (t - burnin) % thinning == 0:
                        samples.append(q_new)

            
            
            if t % check_rate ==0 and t>0 :
                current_acceptance_rate=sum(acceptance_count)/(t+1)
                accept_rates.append(current_acceptance_rate)
                step_sizes.append(step_size)
                log_prob.append(initial_potential)
                if current_acceptance_rate < 0.2:
                    step_size*=0.9
                    acceptance_count=[]
                if current_acceptance_rate > 0.8:
                    step_size*=1.1
                    acceptance_count=[]

            
            
            tr.set_description('HMC')        
            tr.set_postfix(potential=initial_potential, rate=current_acceptance_rate, step=step_size, norm=norm(q_new))
            
            
            
            if np.isnan(q_new).sum():
                print('Current state contains nan')
                break
            q_last=q_new
    return samples, accept_rates, step_sizes, log_prob




