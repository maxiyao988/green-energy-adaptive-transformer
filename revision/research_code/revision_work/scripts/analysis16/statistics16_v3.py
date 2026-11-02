"""Preserve the formal Step9/15 contiguous seed-by-date matrix layout."""
from statistics16_v2 import s
_original=s.distributions
def contiguous_distributions(X,h):return _original(s.np.ascontiguousarray(X,dtype=float),h)
s.distributions=contiguous_distributions
