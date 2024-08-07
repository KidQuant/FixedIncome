"""
A collection of classes corresponding to fixed-income securities, to be fed into (and used with)
other classes and methods available in this repository (eg trees.py, Monte Carlo, etc)
"""


class SelfAmortizingMortgage:
    def __init__(self, dates=mats6, early=False, rate=5.5 / 100, T=6, N=100):
        """Inputs:
        *** early: whether you can prepay the the security when it is more advantageous to do so
        *** T: when the mortgage expires (maturity, years)
        *** N: the face value (dollars)
        *** rate: the interest rate paid yearly
        *** dates: numpy nd array with the payment dates of the security"""
        self.T = T
        self.early = early
        self.rate = rate
        self.date = dates
        # coupon chosen to have value N at time 0, given the mortgage quoted rate
        self.coupon = N / sum(
            [(1 / pow(1 + self.rate, i)) for i in range(1, self.T + 1)]
        )
        if early:
            self._compute_balance(N)

    def get_cf(self, t, r, *args, **kwargs):
        """Inputs:
        *** t: the current time
        *** r: the current interest rate (at that node in the tree)
        Output:
        *** the cash-flow of the security for that (time, interest rate)"""
        return self.coupon

    def exercise_early(self, current_value, val_if_exercise_early, t, r):
        """Method to say whether you should exercise early or not
        Inputs:
        *** current_value: the value of the contract at the node, if you do not exercise early
        *** t: the current time
        *** r: the current interest rate
        Returns: TRUE if should exercise early, FALSE otherwise"""

        if not self.early:
            print(
                "WARNING: this should not be called when the security does not allow for early exercise"
            )
            return False
        return current_value > val_if_exercise_early

    def _compute_balance(self, N):
        """Make a list with the amount of principal left at early possible moment,
        and a list with the interests paid"""
        bal_st = [N]
        interest = []
        princ = []
        for i in range(self.T + 1):  # for each payment until maturity
            interest.append(bal_st[i] * self.rate)
            princ.append(self.coupon - interest[i])
            if i < self.T:
                bal_st.append(bal_st[i] - self.coupon + interest[i])
        self.balances = bal_st
        self.interest = interest
        self.princ = princ

    def cf_early_exercise(self, t, r):
        """Gives the payoff if the security is exercised early, at time t, with rate r"""
        # When a borrower prepays a mortgage, the borrower pays back the remaining principal on
        # the loan, and does not make any of the remaining schedule payments
        return self.balances[t]

    def get_raw_cf_mat(self, interest_rate_tree, *args, **kwargs):
        """Inputs:
        *** interest_rate_tree: a 2D numpy array with corresponding interest rates in each call
        Returns:
        *** the payoff matrix corresponding to the security, a matrix of the same dimension
        """
        payoff_mat = np.zeros(
            (interest_rate_tree.shape[0] + 1, interest_rate_tree.shape[0] + 1)
        )
        for time in reversed(range(1, payoff_mat.shape[0])):
            for node in range(time + 1):
                payoff_mat[node, time] = self.get_cf(
                    time,
                    interest_rate_tree[max(node - 1, 0), time - 1],
                    *args,
                    **kwargs
                )

        return payoff_mat
