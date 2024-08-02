"""
File to compute Black values for securities where a closed-form solution exists (caplets, caps...)
"""

import numpy as np
from scipy.stats import norm

from utils import count_days


def hull_white_caplet(
    sigma,
    kappa,
    discount_factor,
    discount_factor_prev,
    d_settlement,
    strike,
    d_prev,
    d_mat,
    nominal=1,
):
    """
    :param sigma: Hull-White sigma parameter
    :param kappa: Hull-White kappa parameter
    :param discount_factor: discount factor at the time of the caplet pays
    :param discount_factor_prev: discount factor at the d_prev time
    :param d_settlement:
    :param strike:
    :param d_prev: This is the previous, actual payment date
    :param d_mat: This is the actual payment date for the maturity
    :param nominal:
    :return: the caplet val as calculate by the Hull-White model with parameters (kappa, sigma) and given market data
    """
    t_i_prev = count_days(d_prev, d_settlement, "actual") / 360
    t_i = count_days(d_mat, d_settlement, "actual") / 360
    # equilavent to delta in Veronesi (amount of time the caplet is active for). Denoted tau_i in course notes.

    delta = count_days(d_prev, d_mat, method="actual") / 360
    sig_p = (
        sigma
        * np.sqrt((1 - np.exp(-2 * kappa * t_i_prev)) / (2 * kappa))
        * np.abs((1 - np.exp(-kappa * t_i)) / kappa)
    )
    h_i = (1 / sig_p) * (
        np.log(discount_factor * (1 + strike * delta) / discount_factor_prev)
    ) + sig_p / 2
    caplet_val = nominal * (
        discount_factor_prev * norm.cdf(-h_i + sig_p)
        - (1 + strike * delta) * discount_factor * norm.cdf(-h_i)
    )

    return caplet_val


def black_caplet(
    black_vol,
    discount_factor,
    strike,
    forward_libor,
    d_settlement,
    d_prev,
    d_expir,
    d_mat,
    nominal=1,
):
    """
    :param black_vol: Black quoted (implied) vol for this caplet
    :param discount_factor: discount factor applicable
    :param strike:
    :param forward_libor:
    :param d_settlement: settlement date of the contract
    :param d_mat: maturity date of the caplet (denote T_{i + 1} in Veronesi)
    :param d_prev: in the context of a cap, this is the previous date to payment. This serves to determine the length
        of the caplet ("delta" in Veronesi, page 686 - 688)
    :param d_expir: the expiry date (accural expiry date)
    :param norminal:
    :return:
    """
    # equivalent to delta in Veronesi (amount of time the caplet is active for). Denoted tau_i in course notes.
    delta = count_days(d_prev, d_mat, method="actual") / 360
    # this helps determine the volatility (where the expiry date used, d_expir, is the accural date from the contract)
    t_prev_expiry = count_days(d_settlement, d_expir, method="actual") / 365
    vol_t = black_vol * np.sqrt(t_prev_expiry)
    d_1 = (1 / vol_t) * np.log(forward_libor / strike) + 0.5 * vol_t
    d_2 = d_1 - vol_t
    caplet_price = nominal * (
        discount_factor
        * delta
        * (forward_libor * norm.cdf(d_1) - strike * norm.cdf(d_2))
    )
    # print("Delta: {}. T_Prev: {}. Vol_t: {}".format(delta, t_prev_expiry, vol_t))
    return caplet_price
