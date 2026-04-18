import numpy as np


def _hermitian(mat):
    """Numerically symmetrize a Hermitian matrix."""
    return 0.5 * (mat + mat.conj().T)


def _waterfill(lambdas, total_power, eps=1e-12):
    """Classical water-filling over eigenmodes with gains `lambdas`."""
    lambdas = np.asarray(lambdas, dtype=float)
    pos = lambdas > eps
    lam = lambdas[pos]
    p = np.zeros_like(lambdas)

    if lam.size == 0 or total_power <= 0:
        return p, 0.0

    inv_gains = 1.0 / lam
    prefix = np.cumsum(inv_gains)
    k_star, mu = 0, 0.0

    for k in range(1, len(lam) + 1):
        mu_k = (total_power + prefix[k - 1]) / k
        if mu_k > inv_gains[k - 1] + eps:
            k_star, mu = k, mu_k
        else:
            break

    if k_star > 0:
        p_active = np.maximum(0.0, mu - inv_gains)
        p_active[k_star:] = 0.0
        p[pos] = p_active

    return p, mu


def collapse_cir(a_cir, t_idx=0):
    """
    Collapse a Sionna CIR tensor over the path axis.

    Parameters
    ----------
    a_cir : ndarray
        Complex CIR tensor with shape
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps].
    t_idx : int or None
        Time index to extract. If None, returns the full time stack.

    Returns
    -------
    H : ndarray
        If t_idx is None:
            [num_rx * num_rx_ant, num_tx * num_tx_ant, num_time_steps]
        else:
            [num_rx * num_rx_ant, num_tx * num_tx_ant]
    """
    a_cir = np.asarray(a_cir, dtype=complex)
    if a_cir.ndim != 6:
        raise ValueError(
            "a_cir must have 6 dimensions "
            "[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]."
        )

    num_rx, num_rx_ant, num_tx, num_tx_ant, _, num_time_steps = a_cir.shape
    H_t = a_cir.sum(axis=-2)
    H_t = H_t.reshape(num_rx * num_rx_ant, num_tx * num_tx_ant, num_time_steps)

    if t_idx is None:
        return H_t

    t_idx = int(t_idx)
    if not (0 <= t_idx < num_time_steps):
        raise ValueError(f"t_idx={t_idx} is out of range for num_time_steps={num_time_steps}.")
    return H_t[:, :, t_idx]


def dominant_precoder(H0, num_streams=1):
    """
    Dominant right-singular-vector precoder with no power-allocation inputs.

    This is the simple entry when you want a beam direction directly from H0
    without introducing N0/P0. For num_streams > 1, the leading right singular
    vectors are returned with unit-norm columns.
    """
    H0 = np.asarray(H0, dtype=complex)
    if H0.ndim != 2:
        raise ValueError("H0 must be a 2D array of shape (M, Nt).")

    num_streams = int(num_streams)
    if num_streams <= 0:
        raise ValueError("num_streams must be positive.")

    U, s, Vh = np.linalg.svd(H0, full_matrices=False)
    n_keep = min(num_streams, Vh.shape[0])
    W_t = Vh.conj().T[:, :n_keep]
    W_t /= np.linalg.norm(W_t, axis=0, keepdims=True) + 1e-15

    info = {
        "singular_values": s,
        "U": U,
        "Vh": Vh,
        "rank": int(np.count_nonzero(s > 1e-10)),
    }
    return W_t, info


def joint_waterfilling(H0, N0=1.0, P0=1.0):
    """
    Standard joint water-filling with no jammer term.

    Solves:
        max_{Q_t >= 0, Tr(Q_t)=P0} log2 det(I + H0 Q_t H0^H / N0)

    Parameters
    ----------
    H0 : (M, Nt) complex ndarray
        Desired-signal channel seen by the stacked satellite RX array.
    N0 : float
        Noise variance.
    P0 : float
        Desired TX power budget.

    Returns
    -------
    Q_t : (Nt, Nt) complex ndarray
        Optimal transmit covariance.
    info : dict
        Diagnostics, including eigenmodes and power allocation.
    rate_bpcu : float
        Achieved log-det rate in bits/use.
    """
    H0 = np.asarray(H0, dtype=complex)
    if H0.ndim != 2:
        raise ValueError("H0 must be a 2D array of shape (M, Nt).")

    M, _ = H0.shape
    noise_cov = float(N0) * np.eye(M, dtype=complex)
    X = np.linalg.solve(noise_cov, H0)
    Mmat = _hermitian(H0.conj().T @ X)

    evals, V = np.linalg.eigh(Mmat)
    order = np.argsort(evals)[::-1]
    lambdas = np.maximum(evals[order], 0.0)
    V = V[:, order]

    p, mu = _waterfill(lambdas, float(P0))
    Q_t = (V * p) @ V.conj().T

    signal_cov = H0 @ Q_t @ H0.conj().T
    Y = np.linalg.solve(noise_cov, signal_cov)
    I_plus = np.eye(M, dtype=complex) + Y
    sign, logdet = np.linalg.slogdet(I_plus)
    rate_bpcu = (logdet / np.log(2.0)) if sign > 0 else np.nan

    info = {
        "lambdas": lambdas,
        "V": V,
        "p": p,
        "mu": mu,
        "rank": int(np.count_nonzero(p > 1e-10)),
        "rate_bpcu": float(rate_bpcu),
        "noise_cov": noise_cov,
    }
    return Q_t, info, float(rate_bpcu)


def precoder_from_info(info, num_streams=None, tol=1e-10):
    """
    Build a linear precoder matrix W_t from the water-filling solution in `info`.

    Returns W_t such that Q_t = W_t W_t^H over the selected active modes.
    """
    V = np.asarray(info["V"], dtype=complex)
    p = np.asarray(info["p"], dtype=float)
    active = np.flatnonzero(p > tol)

    if num_streams is not None:
        num_streams = int(num_streams)
        if num_streams <= 0:
            raise ValueError("num_streams must be positive.")
        active = active[: min(num_streams, active.size)]

    if active.size == 0:
        return np.zeros((V.shape[0], 0), dtype=complex), active

    W_t = V[:, active] * np.sqrt(p[active])[None, :]
    return W_t, active


def apply_precoder_to_cir(a_cir, W_t, use_precoding=True):
    """
    Apply a TX precoder to a Sionna CIR tensor while preserving per-path delays.

    Parameters
    ----------
    a_cir : ndarray
        Input CIR with shape
        [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps].
    W_t : ndarray
        Precoder with shape [num_tx * num_tx_ant, num_streams].
    use_precoding : bool
        If False, the raw `a_cir` is returned unchanged.

    Returns
    -------
    a_eff : ndarray
        Effective CIR with shape
        [num_rx, num_rx_ant, 1, num_streams, num_paths, num_time_steps].
        The original per-path delays remain valid for the effective channel.
    """
    if not use_precoding or W_t is None:
        return np.asarray(a_cir, dtype=complex)

    a_cir = np.asarray(a_cir, dtype=complex)
    if a_cir.ndim != 6:
        raise ValueError(
            "a_cir must have 6 dimensions "
            "[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]."
        )

    W_t = np.asarray(W_t, dtype=complex)
    if W_t.ndim == 1:
        W_t = W_t.reshape(-1, 1)
    if W_t.ndim != 2:
        raise ValueError("W_t must be a 2D matrix of shape (Nt_total, num_streams).")

    num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps = a_cir.shape
    n_tx_total = num_tx * num_tx_ant
    if W_t.shape[0] != n_tx_total:
        raise ValueError(
            f"W_t expects {W_t.shape[0]} TX dimensions, but a_cir has {n_tx_total}."
        )

    a_flat = a_cir.reshape(num_rx, num_rx_ant, n_tx_total, num_paths, num_time_steps)
    a_eff_flat = np.einsum("urapt,as->urspt", a_flat, W_t, optimize=True)
    num_streams = W_t.shape[1]
    return a_eff_flat.reshape(
        num_rx, num_rx_ant, 1, num_streams, num_paths, num_time_steps
    )


def effective_channel_from_cir(a_cir, W_t=None, t_idx=0, use_precoding=True):
    """Convenience wrapper that returns the collapsed raw or precoded channel matrix."""
    a_used = apply_precoder_to_cir(a_cir, W_t, use_precoding=use_precoding)
    return collapse_cir(a_used, t_idx=t_idx)


def dominant_precoder_from_cir(
    a_tx,
    t_idx=0,
    num_streams=1,
    use_precoding=True,
):
    """
    Solve a dominant-SVD precoder directly from a Sionna CIR tensor.

    This is the lightweight entry when you want a beam direction from `a_tx`
    without passing N0/P0.
    """
    H0 = collapse_cir(a_tx, t_idx=t_idx)
    W_t, info = dominant_precoder(H0, num_streams=num_streams)

    a_eff = apply_precoder_to_cir(a_tx, W_t, use_precoding=True)
    H0_eff = collapse_cir(a_eff, t_idx=t_idx)
    a_use = a_eff if use_precoding else np.asarray(a_tx, dtype=complex)
    H0_use = H0_eff if use_precoding else H0

    result = {
        "H0": H0,
        "info": info,
        "W_t": W_t,
        "w_t": W_t[:, :1] if W_t.shape[1] >= 1 else None,
        "a_eff": a_eff,
        "a_use": a_use,
        "H0_eff": H0_eff,
        "H0_use": H0_use,
        "use_precoding": bool(use_precoding),
        "precoder_type": "dominant",
    }
    return result


def joint_waterfilling_from_cir(
    a_tx,
    N0,
    P0,
    t_idx=0,
    num_streams=None,
    use_precoding=True,
    tol=1e-10,
):
    """
    Solve a joint water-filling precoder directly from a Sionna CIR tensor.

    This is the main helper for your notebooks:
      1. collapse `a_tx` into H0 at snapshot `t_idx`
      2. solve the water-filling covariance
      3. factorize it into a precoder W_t
      4. optionally apply W_t back to `a_tx` to produce `a_eff`
    """
    H0 = collapse_cir(a_tx, t_idx=t_idx)
    Q_t, info, rate_bpcu = joint_waterfilling(H0, N0=N0, P0=P0)
    W_t, active = precoder_from_info(info, num_streams=num_streams, tol=tol)

    a_eff = apply_precoder_to_cir(a_tx, W_t, use_precoding=True)
    H0_eff = collapse_cir(a_eff, t_idx=t_idx)
    a_use = a_eff if use_precoding else np.asarray(a_tx, dtype=complex)
    H0_use = H0_eff if use_precoding else H0

    result = {
        "H0": H0,
        "Q_t": Q_t,
        "info": info,
        "rate_bpcu": rate_bpcu,
        "W_t": W_t,
        "w_t": W_t[:, :1] if W_t.shape[1] >= 1 else None,
        "active_modes": active,
        "a_eff": a_eff,
        "a_use": a_use,
        "H0_eff": H0_eff,
        "H0_use": H0_use,
        "use_precoding": bool(use_precoding),
        "precoder_type": "joint_waterfilling",
    }
    return result
