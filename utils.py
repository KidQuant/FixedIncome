"""
File to perform common conversions between different conventions (rates quoting, day counts...)
"""

import calendar
import datetime
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def iseom(date):
    """given a datetime.date, returns whether that date is at the end of a month"""
    y = date.year
    m = date.month
    return calendar.monthrange(y, m)[1] == date.day


def count_days(date_1, date_2, method="30I/360", T=None):
    """
    :param date_1: datetime.date objects: date_1 and date_2, start date
    :param date_2: end date
    :param method: String: method, desired convention ('30I/360', 'actual')
    :param T: datetime.date object (maturity)
    :return: int: the day count between dates, using given method
    """
    if method == "actual":
        delta = date_2 - date_1
        return abs(delta.days)

    if method == "30I/360":
        y2 = date_2.year
        y1 = date_1.year
        m2 = date_2.month
        m1 = date_1.month
        d2 = date_2.day
        d1 = date_1.day
        if iseom(date_1):  # If D1 is last day of month...
            d1 = 30  # ... change D1 to 30
        # If D2 is last day of month and (date 2 is not maturity or M2 is not February)...
        if iseom(date_2) and (date_2 != T or m2 != 2):
            d2 = 30
        return abs(360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1))

    if method == "30/360":
        # TODO: add special cases here
        y2 = date_2.year
        y1 = date_1.year
        m2 = date_2.month
        m1 = date_1.month
        d2 = date_2.day
        d1 = date_1.day
        return abs(360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1))


def discount_from_zero(date_1, date_2, zero_rate, make_pct=True, met="30I/360", freq=2):
    """
    given the (annual) rate listed and dates between which the rate applies, returns corresponding discount factor
    (implied by the zero rate)
    :param date_1: the settlement date for the given zero (beginning of the zero)
    :param date_2: the maturity date we will have the discount factor for
    :param zero_rate: the corresponding zero rate (double)
    :param make_pct: boolean, whether we give the formula some percents
    :param met: the method for computing days, '30I/360 by default
    :param freq: int, how often compounded / year (by default 2, semiannually-compounded)
    :return: double, the corresponding zero rate
    """
    if make_pct:
        zero_rate /= 100
    num_days = count_days(date_1, date_2, method=met, T=date_2)
    return 1 / (1 + zero_rate / freq) ** (freq * num_days / 360)


def forward_from_discount(discounts):
    """
    :param discounts: Pandas series indices keys being dates, and values being the discount factor that applies
        from date1 to that key
    :return: Pandas series with the forward rate implied by the discount, for each date in index
    """
    fwd = pd.Series()
    for i in range(len(discounts) - 1):
        fwd_rate = (
            (discounts.iloc[i] / discounts.iloc[i + 1] - 1)
            * 360
            / count_days(discounts.index[i], discounts.index[i + 1], method="actual")
            * 100
        )
        to_append = pd.Series(data=[fwd_rate], index=[discounts.index[i]])
        fwd = fwd.append(to_append)
    return fwd
