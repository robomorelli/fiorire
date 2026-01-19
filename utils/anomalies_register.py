
# Import WOMBAT anomaly classes
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# WOMBAT Registry
ANOMALIES_REGISTRY = {
    'GWN':GWN,
    'Constant':Constant,
    'Step':Step,
    'Impulse':Impulse,
    'GNN':GNN,
    'PrincipalSubspaceAlteration': PrincipalSubspaceAlteration,
    }