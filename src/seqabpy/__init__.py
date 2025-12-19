__version__ = "0.2.2"

import numpy as np
from scipy.stats import norm

# np and norm are given to gatsby and gavi via (import *)
from seqabpy import gatsby, gavi

# once everything is imported, with (from seqabpy import *)
# only gavi and gatsby will be imported
__all__ = ["gatsby", "gavi"]
