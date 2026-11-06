# SfcTypeTIModel performance study (EXPERIMENTAL)

Benchmarks, accuracy checks and a showcase notebook for the prototype models
in `oggm/sandbox/sfc_type_fast.py`. The results are discussed in
[`REPORT.md`](REPORT.md).

| File | What it does |
|---|---|
| `common.py` | HEF setup, resampled and synthetic flowlines, calibration and dynamic-run helpers |
| `profile_reference.py` | profile of the reference model (instrumented timers + cProfile) |
| `check_exact.py` | checks that `FastSfcTypeTIModel` reproduces the reference |
| `pytest_fast_plugin.py` | runs the existing OGGM tests with the fast class swapped in |
| `bench_speed.py` | speed benchmarks (kernel, scaling with grid points, HEF workflows, dynamic scaling) |
| `bench_ideas_estimates.py` | kernel-only estimates for ideas that were not implemented |
| `accuracy_fixed_heights.py` | accuracy of `FixedHeightSfcTypeTIModel` for several level spacings |
| `make_figures.py`, `plotstyle.py` | speed figures, shared plot style |
| `sfctype_experimental_models.ipynb` | showcase notebook with analysis plots |
| `results/` | JSON results, cProfile dumps and figures of the runs in the report |

## Running

From the repository root, with OGGM installed in development mode
(`pip install -e .`) and optionally `numba`:

```bash
python benchmarks/sfctype/profile_reference.py
python benchmarks/sfctype/check_exact.py
python benchmarks/sfctype/bench_speed.py          # --quick for a short run
python benchmarks/sfctype/bench_ideas_estimates.py
python benchmarks/sfctype/accuracy_fixed_heights.py
python benchmarks/sfctype/make_figures.py

# existing tests with the fast model (kernel: numba, numpy or auto)
SFC_KERNEL=numba PYTHONPATH=benchmarks/sfctype \
    python -m pytest -p pytest_fast_plugin oggm/tests/test_models.py -k sfc
```

The scripts use the OGGM test sample data (HEF with the HISTALP climate). If
the usual downloads are blocked (e.g. in a sandbox without access to GitHub
archives), the sample data can be built from the git repositories:

```bash
S=/some/tmp/dir
SC=$(python -c "import re,salem;print(re.search(r\"sample_data_gh_commit = '(\w+)'\",open(salem.__file__).read()).group(1))")
git clone -q https://github.com/fmaussion/salem-sample-data.git $S/salem-sd
(cd $S/salem-sd && git checkout -q $SC && mkdir -p ~/.salem_cache && \
  git archive --format=zip --prefix=salem-sample-data-$SC/ -o ~/.salem_cache/salem-sample-data-$SC.zip HEAD)
(cd ~/.salem_cache && python -m zipfile -e salem-sample-data-$SC.zip .)
OC=$(grep -n "^SAMPLE_DATA_COMMIT" oggm/utils/_downloads.py | cut -d"'" -f2)
git clone -q https://github.com/OGGM/oggm-sample-data.git $S/sd
(cd $S/sd && git checkout -q $OC)
D=~/OGGM/download_cache/github.com/OGGM/oggm-sample-data/archive
mkdir -p $D && (cd $S/sd && git archive --format=zip --prefix=oggm-sample-data-$OC/ -o $D/$OC.zip HEAD)
```
