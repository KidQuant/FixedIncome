"""
File to compute Black values for securities where a closed-form solution exists (caplets, caps...)
"""

import numpy as np
from scipy.stats import norm

from utils import count_days
