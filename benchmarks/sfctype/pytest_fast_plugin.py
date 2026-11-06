"""pytest plugin: run the existing tests with FastSfcTypeTIModel (EXPERIMENTAL).

Replaces ``oggm.core.massbalance.SfcTypeTIModel`` by a subclass of
FastSfcTypeTIModel with the same name, so that the existing SfcType tests
exercise the fast implementation.

Usage (from the repository root):

    SFC_KERNEL=numba python -m pytest -p pytest_fast_plugin \\
        oggm/tests/test_models.py -k sfc   # with benchmarks/sfctype in PYTHONPATH
"""
import os


def pytest_configure(config):
    from oggm.core import massbalance, flowline, dynamic_spinup
    from oggm.sandbox.sfc_type_fast import FastSfcTypeTIModel
    kernel = os.environ.get('SFC_KERNEL', 'auto')

    class SfcTypeTIModel(FastSfcTypeTIModel):
        def __init__(self, gdir, fl=None, **kwargs):
            kwargs.setdefault('kernel', kernel)
            super().__init__(gdir, fl=fl, **kwargs)

    SfcTypeTIModel.__qualname__ = 'SfcTypeTIModel'

    # the reference uses super(SfcTypeTIModel, self) with the module global,
    # which must point to the original class while its __init__ runs
    orig = massbalance.SfcTypeTIModel
    orig_init = orig.__init__

    def init(self, *args, **kwargs):
        massbalance.SfcTypeTIModel = orig
        try:
            orig_init(self, *args, **kwargs)
        finally:
            massbalance.SfcTypeTIModel = SfcTypeTIModel
    orig.__init__ = init

    for mod in [massbalance, flowline, dynamic_spinup]:
        if hasattr(mod, 'SfcTypeTIModel'):
            mod.SfcTypeTIModel = SfcTypeTIModel
