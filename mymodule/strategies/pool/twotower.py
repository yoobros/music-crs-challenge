"""Registry shim — real implementation at mymodule.strategies.twotower.pool.

Pool auto-discovery in mymodule/strategies/pool/__init__.py scans this
directory for BasePool subclasses. Keeping TwoTowerPool importable here
preserves the pool name `twotower` (TID `ensemble__twotower__passthrough`)
while the actual model + training code lives alongside at
strategies/twotower/.
"""

from mymodule.strategies.twotower.pool import TwoTowerPool  # noqa: F401
