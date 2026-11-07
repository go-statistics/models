import json
import platform

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm

# W0 smoke case: one small OLS fit on fixed data, to prove the pipeline end
# to end (uv, pinned wheels, python -I on stdin, JSON out). The real cases
# (NIST linear, seeded random designs) arrive with W1.
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = np.array([1.1, 1.9, 3.2, 3.9, 5.1, 6.0])
res = sm.OLS(y, sm.add_constant(x)).fit()

case = {
    "name": "smoke-ols",
    "seed": 0,
    "ints": {"nobs": int(res.nobs), "df_model": int(res.df_model), "df_resid": int(res.df_resid)},
    "floats": {
        "params": [float(v) for v in res.params],
        "bse": [float(v) for v in res.bse],
        "rsquared": [float(res.rsquared)],
        "llf": [float(res.llf)],
    },
}
print(json.dumps({
    "tool": "smoracle",
    "statsmodels": statsmodels.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "machine": platform.machine(),
    "cases": [case],
}))
