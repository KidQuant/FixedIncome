""" File with functions to perform common operation on fixed-income securities
(discounted cash-flow, computing durations, converting between different types of
conventions for quoting interest rates, perform day counts depending on the convention 
chosen...), amortization table...
    Definitions:
* Maturity (T): date at which the last payment to the bondholder is due. The bond is redeemed at this date.
* Coupon (C): interest payment that is made by the issuer to each bondholder, at periodic dates.
* Face Value (N) (or par value): final payment which is made at maturity with the last coupon.
* Principal: Amount borrowed. Generally paid off when the bond matures.
* The Frequency (k) with which coupons are paid (e.g. once every year or once every semester).
* The Coupon Rate (c): by definition, the coupon rate is c = k * C / N
* Price: How much the bond sells for (as opposed to its face value) """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def coupon_freq_mat(c, T, ytm, fv=100, k=1, summary_table=True):
    """
    Purpose: get information about a bond defined by (coupon rate, frequency of compounding,
        maturity, face value)
    :param fv: face value of bond
    :param c: coupon rate
    :param k: compounding freq. If None, is continuous
    :param T: maturity of the bond
    :param ytm: yield to maturity at all horizon (compounded k times per year)
    :param summary_table: true iif want to print table showing computation details
    """
    coupon = c * fv / k  # payments at each period, computed from the coupon
    print("The coupon is ", coupon)
    dcf = 0
    conv = (
        0  # convexity of the bond, ie 2nd derivative of price wrt y, divided by prices
    )
    d_mac = 0  # Macaulay duration
    for i in range(1, int(round(k * T))):  # at each coupon payment period...
        payment_pv = coupon / ((1 + ytm / k) ** i)
        dcf += payment_pv
        d_mac += (i / k) * payment_pv
        conv += (i / k) * (i / k + 1 / k) * payment_pv
    pv_last_payment = (coupon + fv) / (1 + ytm / k) ** (
        T * k
    )  # add the principal payment
    dcf += pv_last_payment
    d_mac += T * pv_last_payment
    d_mac /= dcf
    d_mod = d_mac / (1 + ytm / k)
    conv += T * (T + 1 / k) * pv_last_payment
    conv /= dcf * ((1 + ytm / k) ** 2)
    summary = {
        "DCF": dcf,
        "Macaulay duration": d_mac,
        "Modified duration": d_mod,
        "Dollar duration": d_mod * dcf / 100,
        "Convexity": conv,
        "Dollar convexity": conv * dcf / 100,
    }
    print(pd.DataFrame(summary, index=[0]))
