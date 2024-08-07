""" File to model prepayments. Here, we implement a two-covariate prepayment model following Schwartz and Torous (1989) """

import numpy as np

from utils import *


def lamda_nut(t, p, gamma):
    """
    :param t: time, can be a scalar of a 1D numpy array of times
    :param p, gamma: parameters (we want to fit for)
    :return: an array of same shape as t, where each element is corresponding lambda_nut
    """
    return gamma * p * ((gamma * t) ** (p - 1)) / (1 + (gamma * t) ** p)


def lamda(t, v, beta, p, gamma):
    """
    Hazard rate
    :param t:
    :param v: covariates, can be a
          * 1D np array of shape beta, e.g. [coupon gap (covariate 1), indicator variable for summer months (covariate 2)]
          * 2D np array of shape (num_obs, len(beta)), where each row is an obserbation of the covariates
    :param beta: parameter, 1D np array of shape v, coefficients on the covariates
    :param p, gamma: parameters for lambda_nut
    :return: scalar id v is 1D array, 1D array if v is 2D array (where each element corresponds to the estimated hazard
          rate for that observation)
    """
    lambda_nut = lamda_nut(t, p, gamma)
    lam = lambda_nut * np.exp(v.dot(beta))
    return lam


def survival(t, v, beta, p, gamma):
    """
    :param t:
    :param v: 2D np array of shape (nobs, len(beta))
    :param beta:
    :param p:
    :param gamma:
    :return: The survival  value, 1D array of shape (nobs), where each element is the estimated survival for that obs
    """
    cov_portion = np.exp(v.dot(beta))  # depends on covariates only
    time_portion = np.log(1 + (gamma * t) ** p)
    survival = np.exp(-cov_portion * time_portion)
    return survival


def log_likelihood_schwartz_torous(t, expir_type, v, beta, p, gamma):
    """
    :param t: 1D np.array of times when the loan has expired, or time of the end of the observation if the loan has not
        expired
    :param expir_type: 1D np.array of shape (len(t)), where the element is 1 iif the bond defaults during our observation
    :param v: a np 2D array of shape (num_obs, num_covariates), where each row is a set covariates
    :param beta: parameter to estimate (as before)
    :param p: parameter to estimate (as before)
    :param gamma: parameter to estimate (as before)
    :return: log-likelihood function to estimate a prepayment model as in Schwartz and Tourous
    """
    hazard_part = np.log(lamda(t, v, beta, p, gamma)).dot(expir_type)
    survival_part = np.log(survival(t, v, beta, p, gamma)).sum()
    return hazard_part + survival_part


def log_likelihood_schwartz_torous_gradient(t, expir_type, v, beta, p, gamma):
    """
    :return: the gradient of the log-likelihood of Schwartz-Torous evaluated at the given input
    """

    def del_gamma(t, expir_type, v, beta, p, gamma):
        """Returns the derivative with respect to gamma"""
        obs_part = (
            p / gamma - (t**p) * (p * gamma ** (p - 1)) / (1 + (gamma * t) ** p)
        ).dot(expir_type)
        all_part = (
            -np.exp(v.dot(beta))
            * (t**p)
            * p
            * (gamma ** (p - 1))
            / (1 + (gamma * t) ** p)
        ).sum()
        return obs_part + all_part

    def del_p(t, expir_type, v, beta, p, gamma):
        """Returns the derivative with respect to p"""
        obs_part = (
            (1 / p)
            + np.log(gamma * t)
            - ((gamma * t) ** p) * np.log(gamma * t) / (1 + (gamma * t) ** p)
        ).dot(expir_type)
        all_part = (
            -np.exp(v.dot(beta))
            * ((gamma * t) ** p)
            * np.log(gamma * t)
            / (1 + (gamma * t) ** p)
        ).sum()
        return obs_part + all_part

    def del_beta(t, expir_type, v, beta, p, gamma):
        obs_part = expir_type.dot(v)
        all_part = (-np.exp(v.dot(beta)) * np.log(1 + (gamma * t) ** p)).dot(v)
        return obs_part + all_part

    grad = np.array(del_beta(t, expir_type, v, beta, p, gamma))
    grad = np.append(grad, np.array([del_p(t, expir_type, v, beta, p, gamma)]))
    grad = np.append(grad, np.array([del_gamma(t, expir_type, v, beta, p, gamma)]))
    return grad


def log_survival_dynamic_model(t, t_prev, v, beta, p, gamma):
    """
    :param t: 1D array where each element is the period end time
    :param t_prev: 1D array where each element is the period start time
    :param v:
    :param beta:
    :param p:
    :param gamma:
    :return: The log survival value, 1D array of shape (nobs), where each element is the estimated survival for that obs.
        Each value is the log probability of surviving between time t_prev and time t
    """
    log_s = -np.exp(v.dot(beta)) * (
        np.log(1 + (gamma * t) ** p) - np.log(1 + (gamma * t_prev) ** p)
    )
    return log_s


def log_likelihood_dynamic_model(t, t_prev, expir_type, v, beta, p, gamma):
    """
    NB: It is assumed loans that terminate do not appear in the pool after termination (ie a loan that expires in period
    9 will appear only 9 times in t, t_prev, expir_type, v)
    :param t: 1D array of times of events
    :param t_prev:
    :param expir_type: 1D np.array of shape (len(t)), where the element is 1 iif the bond defaults during our observation
    :param v: a np 2D array of shape (num_obs, num_covariates), where each row is a set covariates
    :param beta:
    :param p:
    :param gamma:
    :return: the log-likelihood function when covariates are time-varying
    """
    assert t.size == expir_type.size
    assert t.size == t_prev.size
    t_expir = t[expir_type == 1]
    v_expir = v[expir_type == 1, :]
    hazard_part = np.log(
        lamda(t_expir, v_expir, beta, p, gamma)
    ).sum()  # get hazard contribution at termination times
    # Add the contribution of the log survival for each of the times prior to the termination
    survival_part = log_survival_dynamic_model(t, t_prev, v, beta, p, gamma).sum()
    return hazard_part + survival_part


def log_likelihood_dynamic_model_gradient(t, t_prev, expir_type, v, beta, p, gamma):
    """
    :return: the gradient of our prepayment model with time-varying covariates
    """
    t_expir = t[expir_type == 1]
    v_expir = v[expir_type == 1, :]

    def del_gamma(t, t_prev, t_expir, v, beta, p, gamma):
        """Compute the derivative wrt gamma (tested with gradient checking)"""
        observed = (
            p / gamma
            - (t_expir**p) * p * (gamma ** (p - 1)) / (1 + (gamma * t_expir) ** p)
        ).sum()
        all_part = -np.exp(v.dot(beta)) * (
            (t**p) * p * (gamma ** (p - 1)) / (1 + (gamma * t) ** p)
            - (t_prev**p) * p * (gamma ** (p - 1)) / (1 + (gamma * t_prev) ** p)
        )
        return observed.sum() + all_part.sum()

    def del_p(t, t_prev, t_expir, v, beta, p, gamma):
        observed = (
            1 / p
            + np.log(gamma)
            + np.log(t_expir)
            - ((gamma * t_expir) ** p)
            * np.log(gamma * t_expir)
            / (1 + (gamma * t_expir) ** p)
        )
        gamma_t = gamma * t.copy()
        gamma_t_prev = gamma * t_prev.copy()
        # terms where t == 0 have a contribution of 0 to the derivative, so set it to 1 to get contribution zero from the log
        gamma_t[gamma_t == 0] = 1
        gamma_t_prev[gamma_t_prev == 0] = 1
        all_part = -np.exp(v.dot(beta)) * (
            (gamma_t**p) * np.log(gamma_t) / (1 + gamma_t**p)
            - (gamma_t_prev**p) * np.log(gamma_t_prev) / (1 + gamma_t_prev**p)
        )
        return observed.sum() + all_part.sum()

    def del_beta(t, t_prev, t_expir, v_expir, v, beta, p, gamma):
        """Compute the derivative wrt beta (tested with gradient checking)"""
        observed = v_expir.sum(axis=0)
        all_part = (
            -np.exp(v.dot(beta))
            * (np.log(1 + (gamma * t) ** p) - np.log(1 + (gamma * t_prev) ** p))
        ).reshape(-1, 1) * v
        return observed + all_part.sum(axis=0)

    grad = np.array(del_beta(t, t_prev, t_expir, v_expir, v, beta, p, gamma))
    grad = np.append(grad, np.array([del_p(t, t_prev, t_expir, v, beta, p, gamma)]))
    grad = np.append(grad, np.array([del_gamma(t, t_prev, t_expir, v, beta, p, gamma)]))
    return grad
