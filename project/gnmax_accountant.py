import math
import numpy as np
import core
import smooth_sensitivity


def default_orders():
    """Return the Renyi orders used by the 2018 plotting scripts."""
    return np.concatenate((
        np.arange(2, 20, .5),
        np.arange(20, 40, .2),
        np.arange(40, 75, .5),
        np.logspace(np.log10(75), np.log10(200), num=20),
    ))


def _smooth_sensitivity_rdp(votes, sigma, orders):
    """RDP cost for releasing the data-dependent epsilon itself."""
    num_teachers = int(np.sum(votes[0]))
    num_classes = votes.shape[1]
    ls_total = np.zeros((len(orders), num_teachers), dtype=float)
    rdp_ss = np.zeros(len(orders), dtype=float)

    is_data_independent = core.is_data_independent_always_opt_gaussian(
        num_teachers, num_classes, sigma, orders)

    for row in votes:
        for order_idx, order in enumerate(orders):
            if is_data_independent[order_idx]:
                continue
            ls_total[order_idx] += (
                smooth_sensitivity.compute_local_sensitivity_bounds_gnmax(
                    row, num_teachers, sigma, order))

    for order_idx, order in enumerate(orders):
        beta = .49 / order
        ss = smooth_sensitivity.compute_discounted_max(beta, ls_total[order_idx])
        if ss <= 0:
            continue
        sigma_ss = ((order * math.exp(2 * beta)) / ss) ** (1 / 3)
        rdp_ss[order_idx] = (
            smooth_sensitivity.compute_rdp_of_smooth_sensitivity_gaussian(
                beta, sigma_ss, order))

    return rdp_ss


def compute_epsilon(votes, sigma, delta, orders=None,
                    include_smooth_sensitivity=True):
  """Return epsilon and optimal Renyi order for clean GNMax vote histograms."""
  votes = np.asarray(votes)

  orders = np.asarray(default_orders() if orders is None else orders,
                      dtype=float)
  rdp = np.zeros(len(orders), dtype=float)
  for row in votes:
    logq = core.compute_logq_gaussian(row, sigma)
    rdp += core.rdp_gaussian(logq, sigma, orders)

  if include_smooth_sensitivity:
    rdp += _smooth_sensitivity_rdp(votes, sigma, orders)

  epsilon, optimal_order = core.compute_eps_from_delta(orders, rdp, delta)
  return {
      'epsilon': float(epsilon),
      'optimal_order': float(optimal_order),
  }
