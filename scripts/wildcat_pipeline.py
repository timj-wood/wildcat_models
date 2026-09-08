#!/usr/bin/env python3
"""
wildcat_pipeline.py -- fit sec_contact and Dennis's wildcat_domestic models to the
                       same Scottish x domestic JSFS and compare them with CLAIC.

The Scottish population is the 16 WILD-CAUGHT cats only. The 10 captive-bred
cats are parsed and available as a separate population but are not analysed. 
Hudson FST between the two Scottish groups is 0.085, so pooling them was not innocuous.

Results go to results_wild/, NOT results/, so the earlier pooled-sample fits
survive untouched. Do not compare log-likelihoods across the two directories:
they are computed from different spectra and are not on the same scale.

Four stages:

    python wildcat_pipeline.py sfs                     # MSMC -> data.fs + bootstraps  (once, ~5 min)
    python wildcat_pipeline.py fit --model sec_contact # one restart                   (SLURM array)
    python wildcat_pipeline.py report                  # CSVs + CLAIC comparison       (once)
    python wildcat_pipeline.py plot                    # four-panel fit figure         (report does this too)

Run every stage through sbatch -- see submit_sfs.sh, submit.sh, submit_report.sh.
Nothing here belongs on a login node.

Everything goes through documented dadi entry points:

    dadi.Spectrum.from_data_dict          build the spectrum (project, then fold)
    dadi.Misc.fragment_data_dict          split the genome into blocks
    dadi.Misc.bootstraps_from_dd_chunks   block bootstrap replicates
    dadi.Inference.optimize_log_fmin      unconstrained fit (sec_contact)
    dadi.Inference.opt + nlopt.LN_COBYLA  constrained fit (Dennis's models need TA > max(TB,TD))
    dadi.Godambe.get_godambe              via claic.claic, for CLAIC and for the
                                          confidence intervals, which reuse the
                                          same J and H rather than calling
                                          GIM_uncert and recomputing them

Requires, in the same directory: wildcat_models.py and claic.py (both unmodified).
Written against dadi 2.4.4.
"""

import argparse
import csv
import glob
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import nlopt
import dadi

import claic
import wildcat_models as W

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILES = ["msmc_input_chr1.txt", "msmc_input_chr2.txt"]

# Haplotype positions within column 4 of the MSMC file. 46 diploid cats = 92
# haplotypes; individual i occupies positions 2i and 2i+1. Verified by PCA and
# genotype concordance, not taken from the MSMC documentation.
#
# Boundaries follow the individual order in cat_orders.docx: 16 Scottish wild,
# 10 Scottish captive, 6 domestic, 14 European wild. Checked against the data --
# Hudson FST between columns 0-31 and 32-51 is 0.0850, matching the wild/captive
# figure measured independently, so the split falls where the file says it does.
POPS = {
    "SCO_WILD": list(range(0, 32)),    # 16 wild-caught Scottish wildcats
    "SCO_CAPT": list(range(32, 52)),   # 10 captive-bred Scottish wildcats
    "DOM": list(range(52, 64)),        #  6 domestic cats
    "EUR": list(range(64, 92)),        # 14 mainland European wildcats
}

# Analysed pair, in order. pop1 = Scottish wildcat, pop2 = domestic, which is the
# order both models assume. m12 is migration INTO pop1 FROM pop2.
#
# SCO_CAPT and EUR are parsed but not analysed. They stay in POPS so that the
# column mapping is documented in one place and so that swapping the analysed
# pair is a one-line change; carrying them costs nothing, since from_data_dict
# takes only the populations named here.
ANALYSIS_POPS = ("SCO_WILD", "DOM")

MU = 0.86e-8               # per bp per generation
GENERATION_TIME = 3.0      # years

N_BOOT = 100               # block bootstrap replicates, for CLAIC
CHUNK_SIZE = 1_000_000     # block size in bp
BOOT_SEED = 20260807

GRID_PAD = (10, 20, 30)    # grid = max(sample size) + each of these

# Deliberately NOT "results": that directory holds the pooled-Scottish fits, and
# stage_report's --use-cached would otherwise pick up claic_*.pkl written against
# a 53 x 13 spectrum and rank them against 33 x 13 fits. Composite log-likelihoods
# are not comparable across spectra, so the resulting dCLAIC would be meaningless
# and would look perfectly reasonable. Separate directory, separate run.
#
# Read from the environment so that the submission scripts, which have to build
# the same paths in shell, cannot drift from this value. Set WILDCAT_OUTDIR in
# one place and both agree.
OUTDIR = os.environ.get("WILDCAT_OUTDIR", "results_wild")
FS_FILE = os.path.join(OUTDIR, "data.fs")
BOOT_GLOB = os.path.join(OUTDIR, "boots", "boot_*.fs")
META_FILE = os.path.join(OUTDIR, "meta.json")


# =============================================================================
# PART 1: MSMC -> dadi data dictionary
# =============================================================================

def parse_msmc(filenames):
    """
    Read MSMC multihetsep files into a dadi data dictionary.

    File format (tab separated, one line per segregating site):
        col 0  chromosome
        col 1  position
        col 2  number of callable sites since the previous line
        col 3  string of 92 phased bases

    The data dictionary is dadi's documented input format (see
    `dadi.Misc.make_data_dict`), keyed 'chromosome_position' so that
    `dadi.Misc.fragment_data_dict` can split it into genomic blocks later.

    Sites are entered unpolarised: 'segregating' holds the two observed bases and
    'calls' the count of each per population. With no 'outgroup_allele' key,
    `count_data_dict` treats the first base as ancestral and marks the site
    unpolarised, so `from_data_dict(..., polarized=False)` folds the spectrum at
    the end. Which base is called first is therefore irrelevant.

    Sites with more than two alleles break the fold's (i,j) <-> (n1-i, n2-j)
    symmetry, so they are counted and dropped.

    Returns (data_dict, L, stats), L being the total callable sites.
    """
    dd = {}
    L = 0
    n_multi = n_malformed = 0
    idx = {pop: np.array(v, dtype=np.intp) for pop, v in POPS.items()}

    for fn in filenames:
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        L_file = n_file = multi_file = 0
        with open(fn) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 4:
                    n_malformed += 1
                    continue
                chrom, pos, ncall, g = parts[0], parts[1], parts[2], parts[3].upper()
                try:
                    L_file += int(ncall)
                except ValueError:
                    n_malformed += 1
                    continue
                if len(g) != sum(len(v) for v in POPS.values()):
                    n_malformed += 1
                    continue

                bases = np.frombuffer(g.encode("ascii"), dtype=np.uint8)
                vals = np.unique(bases)
                if len(vals) > 2:
                    multi_file += 1
                    continue
                if len(vals) < 2:
                    continue                      # monomorphic across all 92

                calls = {}
                for pop, ii in idx.items():
                    b = bases[ii]
                    n1 = int((b == vals[0]).sum())
                    calls[pop] = (n1, len(ii) - n1)

                dd["{}_{}".format(chrom, pos)] = {
                    "segregating": (chr(vals[0]), chr(vals[1])),
                    "calls": calls,
                }
                n_file += 1

        L += L_file
        n_multi += multi_file
        print("  {}: {:,} biallelic, {:,} multiallelic dropped, L = {:,}"
              .format(fn, n_file, multi_file, L_file), flush=True)

    stats = {"n_sites": len(dd), "n_multiallelic": n_multi,
             "n_malformed": n_malformed, "L": L}
    return dd, L, stats


# =============================================================================
# PART 2: MODELS
# =============================================================================
#
# Both models return a FOLDED spectrum, to match the folded data, with pop_ids
# set to the analysed pair. Folding and Richardson extrapolation are both linear,
# so folding inside the model function is equivalent to folding afterwards.
# Setting pop_ids stops dadi warning on every likelihood evaluation, since
# Dennis's models label their output 'wildcat'/'domestic'.

def _fold(func):
    def wrapped(params, ns, pts):
        fs = func(params, ns, pts).fold()
        fs.pop_ids = list(ANALYSIS_POPS)
        return fs
    wrapped.__name__ = func.__name__
    return wrapped


def _counted(func):
    """
    Count calls to a model function, to work out why an optimiser stopped.

    dadi.Inference.opt returns only (xopt, fopt); the nlopt result code is not
    exposed, so the stopping criterion has to be inferred. Comparing the
    evaluations and wall time used per round against maxeval and maxtime
    distinguishes 'hit the evaluation limit' and 'hit the time limit' from
    'triggered ftol_abs', which are three different problems with three
    different fixes.
    """
    def wrapped(*a, **kw):
        wrapped.n += 1
        return func(*a, **kw)
    wrapped.n = 0
    return wrapped


def _dennis_spec(func, min_mig=1e-4, constrain_times=True):
    """
    Bounds, starting values and the TA > max(TB, TD) constraint, from Dennis.

    One deviation: his migration lower bounds are exactly 0, and COBYLA raises
    nlopt.RoundoffLimited when a parameter is pinned against a zero bound. The
    floor here is negligible biologically -- at Nref ~ 5e4, M = 1e-4 is
    m ~ 1e-9 per generation, i.e. one migrant every hundred million cats -- so it
    removes the numerical failure without removing "no gene flow" from the model.
    Pass min_mig=0 to restore his bounds exactly.

    Pass constrain_times=False for a model parameterised in durations, where the
    ordering is implied by positivity. It has to be an argument rather than
    something overridden in the returned dict, because
    make_time_ordering_constraint looks up 'TA', 'TB' and 'TD' by name and
    raises on a model that has none of them.
    """
    names, p0, lower, upper = W.model_defaults(func)
    lower = [max(lo, min_mig) if n.startswith("m") else lo
             for n, lo in zip(names, lower)]
    constraint = (W.make_time_ordering_constraint(func)
                  if constrain_times else None)
    return dict(func=_fold(func), names=names, p0=p0, lower=lower, upper=upper,
                constraint=constraint)


# Parameter descriptions and rescaling rules, per model.
#   divergence  parameters whose SUM is the total divergence time. Dennis's times
#               are absolute before present, so his total is TA alone; sec_contact's
#               T1 and T2 are epoch durations and must be added.
#   migrants    (migration parameter, size of the RECEIVING population at that time)

MODELS = {
    "sec_contact": dict(
        func=_fold(dadi.Demographics2D.sec_contact_asym_mig),
        names=["nu1", "nu2", "m12", "m21", "T1", "T2"],
        p0=[1.0, 1.0, 0.5, 0.5, 1.0, 0.5],
        lower=[1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4],
        upper=[100, 100, 50, 50, 20, 20],
        constraint=None,
        divergence=["T1", "T2"],
        migrants=[("m12", "nu1"), ("m21", "nu2")],
        desc={
            "nu1": "Relative size of Scottish wild-caught population",
            "nu2": "Relative size of domestic population",
            "m12": "Migration domestic->Scottish wild (2*Nref*m)",
            "m21": "Migration Scottish wild->domestic (2*Nref*m)",
            "T1": "Duration of isolation (2*Nref generations)",
            "T2": "Duration of secondary contact (2*Nref generations)",
        },
    ),
}

# Models backed by a function in wildcat_models.py, as
# (name, function name, constrain_times, metadata).
#
# Registered only if the function actually exists. A model that cannot be built
# is skipped and named in the log rather than taking the whole module down --
# referencing a missing W.<func> in the dict literal raised AttributeError at
# import, which killed every stage, including `sfs`, which uses no model
# function at all.
_DENNIS_MODELS = [
    ("basic", "wildcat_domestic", True, dict(
        divergence=["TA"],
        migrants=[("m_ls", "nuS"), ("m_sl", "nuL"),
                  ("m2_ds", "nuB"), ("m2_sd", "nuD")],
        desc={
            "nuS": "Silvestris branch after the split",
            "nuL": "Lybica branch after the split",
            "nuB": "Scottish wildcat from TB to the present",
            "nuD": "Domestic cat from TD to the present",
            "TA": "Split of silvestris and lybica (before present)",
            "TB": "Wildcat size change / landbridge (before present)",
            "TD": "Domestic size change / domestication (before present)",
            "m_ls": "Early gene flow into silvestris from lybica (2*Nref*m)",
            "m_sl": "Early gene flow into lybica from silvestris (2*Nref*m)",
            "m2_ds": "Recent gene flow into wildcat from domestic (2*Nref*m)",
            "m2_sd": "Recent gene flow into domestic from wildcat (2*Nref*m)",
        },
    )),
    ("growth", "wildcat_domestic_growth", True, dict(
        divergence=["TA"],
        migrants=[("m_ls", "nuS"), ("m_sl", "nuL"),
                  ("m2_ds", "nuBcurr"), ("m2_sd", "nuDcurr")],
        desc={
            "nuS": "Silvestris branch immediately after the split",
            "nuL": "Lybica branch immediately after the split",
            "nuB": "Wildcat at TB, reached exponentially from nuS",
            "nuD": "Domestic at TD, reached exponentially from nuL",
            "nuBcurr": "Present-day wildcat size",
            "nuDcurr": "Present-day domestic size",
            "TA": "Split of silvestris and lybica (before present)",
            "TB": "Wildcat size change / landbridge (before present)",
            "TD": "Domestic size change / domestication (before present)",
            "m_ls": "Early gene flow into silvestris from lybica (2*Nref*m)",
            "m_sl": "Early gene flow into lybica from silvestris (2*Nref*m)",
            "m2_ds": "Recent gene flow into wildcat from domestic (2*Nref*m)",
            "m2_sd": "Recent gene flow into domestic from wildcat (2*Nref*m)",
        },
    )),
]

UNAVAILABLE_MODELS = {}
for _name, _funcname, _constrain, _meta in _DENNIS_MODELS:
    _func = getattr(W, _funcname, None)
    if _func is None:
        UNAVAILABLE_MODELS[_name] = "wildcat_models.{} is not defined".format(_funcname)
        continue
    try:
        MODELS[_name] = dict(_meta,
                             **_dennis_spec(_func, constrain_times=_constrain))
    except Exception as _e:
        # model_defaults raises if the p0/bounds tables have no entry for the
        # function -- the mirror image of the missing-def case.
        UNAVAILABLE_MODELS[_name] = repr(_e)

if UNAVAILABLE_MODELS:
    print("wildcat_pipeline: {} model(s) unavailable and not registered:".format(
        len(UNAVAILABLE_MODELS)), file=sys.stderr)
    for _name, _why in UNAVAILABLE_MODELS.items():
        print("  {}: {}".format(_name, _why), file=sys.stderr)

if not MODELS:
    sys.exit("no models could be built; check wildcat_models.py")


def _available(names):
    """Argparse defaults, filtered to models that actually registered."""
    return [n for n in names if n in MODELS]


def grid_points(ns):
    return [int(max(ns)) + p for p in GRID_PAD]


# =============================================================================
# PART 3: ONE OPTIMISATION RESTART (the SLURM array unit)
# =============================================================================

def run_one_restart(model_name, fs, seed, maxiter=300, maxeval=2000, ftol=1e-2,
                    maxtime=1800.0, p0_override=None, fold=1.0, verbose=False):
    """
    One restart from a perturbed starting point, run twice.

    Both optimisers stall short of the optimum on the first pass -- Nelder-Mead
    collapses onto a degenerate simplex, COBYLA shrinks its trust region -- so
    each restart is run in two rounds, the second seeded from the first. This is
    what turns 'stopped on the iteration limit' into a converged fit.

    Unconstrained models use dadi.Inference.optimize_log_fmin, which searches in
    log space and so copes with parameters spanning several orders of magnitude.
    Constrained models must use dadi.Inference.opt with nlopt.LN_COBYLA, the only
    documented route to ineq_constraints. Note that opt's own log_opt=True is
    unusable in dadi 2.4.4: it returns exp of the STARTING point rather than the
    optimum (Inference.py, `if log_opt: xopt = np.exp(p0)`).

    Pass p0_override (from the 'best' stage) and a shrinking fold to run staged
    rounds instead of flat restarts: round 1 explores widely from P0, each later
    round perturbs less around the best point found so far. Round-to-round this
    concentrates the search where it is worth searching, which is usually worth
    more than the same compute spent on more independent restarts.

    Failure returns success=False rather than raising, so one bad array task does
    not lose the batch.
    """
    spec = MODELS[model_name]
    names, p0, lb, ub = spec["names"], list(spec["p0"]), spec["lower"], spec["upper"]
    if p0_override is not None:
        p0 = list(p0_override)
        if len(p0) != len(names):
            raise ValueError("start point has {} values, {} needs {}".format(
                len(p0), model_name, len(names)))
    pts_l = grid_points(fs.sample_sizes)
    func_ex = _counted(dadi.Numerics.make_extrap_log_func(spec["func"]))

    # dadi.Misc.perturb_params uses numpy's LEGACY global RandomState.
    # np.random.default_rng(seed) is silently ignored here.
    np.random.seed(seed)
    p_start = dadi.Misc.perturb_params(p0, fold=fold, lower_bound=lb, upper_bound=ub)

    if spec["constraint"] is not None:
        # Start inside the feasible region: push TA above both event times.
        iTA, iTB, iTD = (names.index(k) for k in ("TA", "TB", "TD"))
        p_start[iTA] = min(max(p_start[iTA], 1.05 * max(p_start[iTB], p_start[iTD])),
                           ub[iTA])

    t0 = time.time()
    out = {"model": model_name, "seed": seed, "param_names": names,
           "p_start": np.asarray(p_start, dtype=float), "pts_l": pts_l,
           "fold": fold, "seeded_from_best": p0_override is not None,
           "sample_sizes": tuple(int(x) for x in fs.sample_sizes)}

    rounds_diag = []
    try:
        p = list(p_start)
        for _ in range(2):
            n0, t_round = func_ex.n, time.time()
            if spec["constraint"] is None:
                p = dadi.Inference.optimize_log_fmin(
                    p, fs, func_ex, pts_l, lower_bound=lb, upper_bound=ub,
                    maxiter=maxiter, verbose=int(verbose))
            else:
                # ftol_abs is ABSOLUTE, on a log-likelihood of order 1e3, and
                # dadi's default of 1e-6 is therefore unreachable: COBYLA never
                # triggers it and always runs to maxeval. Restarts only have to
                # find the right basin -- report polishes the winner to full
                # precision before CLAIC -- so a looser tolerance here costs
                # nothing and stops the optimiser grinding once it has converged.
                p, _ll = dadi.Inference.opt(
                    p, fs, func_ex, pts_l, lower_bound=lb, upper_bound=ub,
                    ineq_constraints=[(spec["constraint"], 1e-6)],
                    algorithm=nlopt.LN_COBYLA, maxeval=maxeval,
                    ftol_abs=ftol, maxtime=maxtime, verbose=int(verbose))
            rounds_diag.append({"n_eval": func_ex.n - n0,
                                "seconds": time.time() - t_round})
        p = np.asarray(p, dtype=float)

        model_fs = func_ex(p, fs.sample_sizes, pts_l)
        ll = float(dadi.Inference.ll_multinom(model_fs, fs))
        theta = float(dadi.Inference.optimal_sfs_scaling(model_fs, fs))
        out.update({
            "success": bool(np.isfinite(ll)), "params": p, "ll": ll, "theta": theta,
            "n_params": len(names), "rounds": rounds_diag,
            "at_bound": [bool(np.isclose(v, lo, rtol=1e-3) or np.isclose(v, hi, rtol=1e-3))
                         for v, lo, hi in zip(p, lb, ub)],
        })
    except Exception as e:
        out.update({"success": False, "error": repr(e), "params": None,
                    "ll": -np.inf, "theta": np.nan, "rounds": rounds_diag})

    out["seconds"] = time.time() - t0
    return out


def polish_fit(model_name, fs, params, maxiter=200, maxeval=2000, rounds=3,
               tol=1e-4, maxtime=3600.0, ftol=1e-4):
    """
    Re-optimise from the best restart until the likelihood stops moving.

    CLAIC is built from second derivatives of the likelihood at p0, so p0 has to
    sit at the maximum to more precision than the finite-difference step can
    resolve. It does not take much to break this: at a point 0.007 log-likelihood
    units short of the optimum, tr(J.H^-1) drifted from 283 to 222 as eps went
    0.02 -> 0.001, i.e. no stable value at all. Polished to convergence, the same
    quantity moved only from 257 to 249, which is inside the noise floor.

    So this runs before the Godambe matrices, always. It is cheap next to the
    hundreds of restarts that produced the fit.
    """
    spec = MODELS[model_name]
    lb, ub = spec["lower"], spec["upper"]
    func_ex = dadi.Numerics.make_extrap_log_func(spec["func"])
    pts_l = grid_points(fs.sample_sizes)

    p = list(np.asarray(params, dtype=float))
    ll = float(dadi.Inference.ll_multinom(func_ex(p, fs.sample_sizes, pts_l), fs))
    ll_start = ll

    for rnd in range(rounds):
        t_round = time.time()
        try:
            if spec["constraint"] is None:
                p_new = dadi.Inference.optimize_log_fmin(
                    p, fs, func_ex, pts_l, lower_bound=lb, upper_bound=ub,
                    maxiter=maxiter)
            else:
                # Capped: an uncapped COBYLA polish on the 11-parameter model ran
                # past a 6 h wall clock without returning. maxtime makes it hand
                # back its best point instead of being killed with nothing.
                # ftol_abs is absolute on a log-likelihood of order 1e3, so
                # dadi's 1e-6 default is unreachable and COBYLA burns the full
                # maxeval on every round even when it converged on the first
                # few evaluations. 1e-4 is still 70x tighter than the 7e-3
                # offset that was enough to destabilise the CLAIC trace.
                p_new, _ = dadi.Inference.opt(
                    p, fs, func_ex, pts_l, lower_bound=lb, upper_bound=ub,
                    ineq_constraints=[(spec["constraint"], 1e-6)],
                    algorithm=nlopt.LN_COBYLA, maxeval=maxeval, maxtime=maxtime,
                    ftol_abs=ftol)
        except Exception as e:
            print("  polish failed ({}), keeping the unpolished fit".format(e))
            break
        ll_new = float(dadi.Inference.ll_multinom(
            func_ex(p_new, fs.sample_sizes, pts_l), fs))
        if not np.isfinite(ll_new) or ll_new <= ll:
            break
        gain, p, ll = ll_new - ll, list(np.asarray(p_new, dtype=float)), ll_new
        print("  polish round {}: ll {:.4f} (gain {:.2e}, {:.0f}s)".format(
            rnd, ll, gain, time.time() - t_round), flush=True)
        if gain < tol:
            break

    theta = float(dadi.Inference.optimal_sfs_scaling(
        func_ex(p, fs.sample_sizes, pts_l), fs))
    return p, ll, theta, ll - ll_start


def parameter_errors(res, params, theta):
    """
    Per-parameter standard errors from the same J and H that produced the CLAIC.

    Not dadi.Godambe.GIM_uncert, which would recompute both matrices from the
    100 replicates a third time -- basic is the expensive one. Reusing res also
    means the intervals and the CLAIC refer to the same matrices at the same
    step size rather than to a second, independently computed pair.

    The Godambe matrix is G = H J^-1 H, so the sandwich covariance G^-1 is
    H^-1 J H^-1. That form needs only H inverted, instead of building G and
    re-inverting it, which matters on a matrix whose condition number is ~1e15.

    Errors come back on the LOG scale, by the delta method, SE(log x) =
    SE(x) / x, so that the interval x * exp(+-1.96 SE) stays positive. TB's
    linear-scale lower limit is negative, which is not a time. Differencing in
    log space directly is not the alternative: it puts a negative eigenvalue
    into H on this data (see the note in stage_report).

    theta is appended last, matching claic's multinom=True convention, so the
    returned array has k+1 entries in the same order as the effective parameter
    count is read.

    A negative sandwich variance is a flat direction in H rather than an error,
    and is returned as nan rather than clipped, so that the caller can report
    the interval as undefined instead of inventing one.
    """
    J, H = np.asarray(res["J"]), np.asarray(res["H"])
    Hinv = np.linalg.inv(H)
    var = np.diag(Hinv @ J @ Hinv)
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.where(var > 0, var, np.nan))
    vals = np.append(np.asarray(params, dtype=float), theta)
    if len(se) != len(vals):
        # multinom=False, or some other convention: theta is not in the matrix.
        vals = vals[:len(se)]
    return se / vals


# =============================================================================
# PART 4: DADI UNITS -> BIOLOGICAL UNITS
# =============================================================================

def rescale(model_name, params, theta, L, mu=MU, gen_time=GENERATION_TIME):
    """
    Convert coalescent units to cats, generations, years and migrants.

        theta = 4 * Nref * mu * L   ->  Nref = theta / (4 * mu * L)
        nu                          ->  N = nu * Nref
        T (units of 2*Nref gens)    ->  generations = T * 2 * Nref
        m12 (= 2*Nref*m)            ->  m per generation = m12 / (2 * Nref)

    Migrant INDIVIDUALS entering a population each generation is
        N_receiving * m = (nu * Nref) * (M / (2 * Nref)) = nu * M / 2,
    which scales with the receiving population, not with Nref. Writing M/2
    instead gives Nref*m, a much larger and quite different quantity whenever the
    receiving population is small -- exactly the Scottish case.

    Nref is the ancestral population in both models (dadi's phi_1D starts at size
    1), so the two are directly comparable, but they need not come out equal: the
    models put drift in different places.
    """
    spec = MODELS[model_name]
    p = dict(zip(spec["names"], np.asarray(params, dtype=float)))
    Nref = theta / (4.0 * mu * L)

    out = {"Nref": Nref, "sizes": {}, "times": {}, "rates": {}, "migrants": {}}
    for name, v in p.items():
        if name.startswith("nu"):
            out["sizes"][name] = v * Nref
        elif name.startswith("T"):
            out["times"][name] = (v * 2.0 * Nref, v * 2.0 * Nref * gen_time)
        elif name.startswith("m"):
            out["rates"][name] = v / (2.0 * Nref)

    for mig, size in spec["migrants"]:
        if mig in p and size in p:
            out["migrants"][mig] = p[size] * p[mig] / 2.0

    T_div = sum(p[k] for k in spec["divergence"])
    out["divergence"] = (T_div, T_div * 2.0 * Nref, T_div * 2.0 * Nref * gen_time)
    out["divergence_from"] = " + ".join(spec["divergence"])
    return out


# =============================================================================
# STAGE 1: SPECTRUM AND BOOTSTRAPS
# =============================================================================

def stage_sfs(args):
    os.makedirs(os.path.join(OUTDIR, "boots"), exist_ok=True)
    pops = list(ANALYSIS_POPS)
    ns = [len(POPS[p]) for p in pops]

    print("=== parsing MSMC input ===", flush=True)
    dd, L, stats = parse_msmc(args.input)
    print("  total: {:,} biallelic sites, L = {:,}".format(stats["n_sites"], L))

    # polarized=False: dadi projects to `ns` and then folds, in that order.
    # Folding before projection would be wrong; from_data_dict gets it right.
    fs = dadi.Spectrum.from_data_dict(dd, pops, ns, polarized=False)
    fs.to_file(FS_FILE)
    print("\n=== folded JSFS: {} x {} ===".format(*pops))
    print("  array shape   : {}   (ns = {}, so bins are 0..{} and 0..{})"
          .format(fs.shape, tuple(int(n) for n in fs.sample_sizes), ns[0], ns[1]))
    print("  folded        : {}".format(fs.folded))
    print("  segregating   : {:.0f}".format(fs.S()))
    print("  wrote {}".format(FS_FILE), flush=True)

    print("\n=== block bootstraps ===", flush=True)
    chunks = dadi.Misc.fragment_data_dict(dd, args.chunk)
    print("  {} blocks of {:,} bp".format(len(chunks), args.chunk))

    # The Godambe J is a k x k covariance estimated across BLOCKS, not across
    # bootstrap replicates -- resampling 100 times from 22 blocks does not create
    # information that 22 blocks do not contain. Once k approaches the block
    # count, J is near-singular and the CLAIC is unstable no matter how well the
    # optimiser converged.
    k_max = max(len(spec["names"]) for spec in MODELS.values())
    if len(chunks) < 3 * k_max:
        print("  NOTE: {} blocks vs {} parameters in the widest model. Expect an "
              "unstable CLAIC there; --chunk 500000 roughly doubles the block "
              "count (LD in cats decays well inside 500 kb, so the blocks stay "
              "approximately independent)."
              .format(len(chunks), k_max), flush=True)
    random.seed(args.seed)          # bootstraps_from_dd_chunks uses random.choices
    boots = dadi.Misc.bootstraps_from_dd_chunks(chunks, args.nboot, pops, ns,
                                                polarized=False)
    for i, b in enumerate(boots):
        if not np.array_equal(b.mask, fs.mask):
            raise RuntimeError("bootstrap {} has a different mask from the data"
                               .format(i))
        b.to_file(os.path.join(OUTDIR, "boots", "boot_{:03d}.fs".format(i)))
    sizes = [float(b.sum()) for b in boots]
    print("  {} replicates, mean total {:,.0f} vs data {:,.0f}"
          .format(len(boots), np.mean(sizes), float(fs.sum())))
    print("  wrote {}".format(os.path.join(OUTDIR, "boots")), flush=True)

    meta = {"pops": pops, "ns": ns, "L": L, "mu": MU,
            "generation_time": GENERATION_TIME, "chunk_size": args.chunk,
            "n_boot": args.nboot, "boot_seed": args.seed,
            "input_files": list(args.input), "pts_l": grid_points(ns), **stats}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote {}".format(META_FILE))


# =============================================================================
# STAGE 2: FIT
# =============================================================================

def stage_fit(args):
    os.makedirs(OUTDIR, exist_ok=True)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", args.task_id))
    fs = dadi.Spectrum.from_file(FS_FILE)

    print("model={}  task={}  ns={}  pts={}".format(
        args.model, task_id, tuple(fs.sample_sizes), grid_points(fs.sample_sizes)),
        flush=True)

    p0_override = None
    if args.start_from:
        with open(args.start_from) as f:
            seed_json = json.load(f)
        p0_override = seed_json["params"]
        print("  seeded from {} (ll={:.4f}), fold={}".format(
            args.start_from, seed_json.get("ll", float("nan")), args.fold),
            flush=True)

    fn = os.path.join(OUTDIR, "fit_{}_task{:04d}.pkl".format(args.model, task_id))
    results = []
    for k in range(args.restarts):
        r = run_one_restart(args.model, fs, seed=(args.seed_base + task_id) * 1000 + k,
                            maxiter=args.maxiter, maxeval=args.maxeval,
                            ftol=args.ftol, maxtime=args.maxtime,
                            p0_override=p0_override, fold=args.fold,
                            verbose=args.verbose)
        r["task_id"] = task_id
        results.append(r)
        # Written after every restart, so a wall-clock kill loses only the
        # restart in flight rather than the whole task.
        with open(fn, "wb") as f:
            pickle.dump(results, f)
        if r["success"]:
            print("  restart {}: ll={:.3f}  theta={:.1f}  ({:.0f}s)".format(
                k, r["ll"], r["theta"], r["seconds"]), flush=True)
        else:
            print("  restart {}: FAILED {}".format(k, r.get("error")), flush=True)
    print("wrote {}".format(fn), flush=True)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_fit(model_name, fs, params, ll, path, resid_range=None):
    """
    dadi's own four-panel comparison: data, model, residuals, residual histogram.

    plot_2d_comp_multinom rescales the model to the data itself, so the raw
    model spectrum goes in. Agg is selected here rather than at import so that
    the module still loads on a machine with no display -- every compute node.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts_l = grid_points(fs.sample_sizes)
    func_ex = dadi.Numerics.make_extrap_log_func(MODELS[model_name]["func"])
    model = func_ex(params, fs.sample_sizes, pts_l)

    plt.close("all")
    # vmin=1 keeps the log colour scale off the empty bins of a sparse SFS.
    dadi.Plotting.plot_2d_comp_multinom(model, fs, vmin=1,
                                        pop_ids=list(ANALYSIS_POPS),
                                        resid_range=resid_range, show=False)
    fig = plt.gcf()
    fig.suptitle("{}   ll = {:.2f}".format(model_name, ll))
    # dadi's default spacing runs the lower titles into the upper axis labels.
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def stage_plot(args):
    fs = dadi.Spectrum.from_file(FS_FILE)
    for name in args.models:
        results, ok = gather(name)
        if not ok:
            print("no successful fits for {}, skipping".format(name))
            continue
        best = ok[0]
        path = os.path.join(OUTDIR, "fit_{}.png".format(name))
        plot_fit(name, fs, best["params"], best["ll"], path,
                 resid_range=args.resid_range)
        print("wrote {}".format(path))


# =============================================================================
# STAGE 3: REPORT
# =============================================================================

def gather(model_name):
    files = sorted(glob.glob(os.path.join(OUTDIR, "fit_{}_task*.pkl".format(model_name))))
    results = []
    for fn in files:
        with open(fn, "rb") as f:
            results.extend(pickle.load(f))
    ok = [r for r in results if r.get("success") and np.isfinite(r["ll"])]
    ok.sort(key=lambda r: r["ll"], reverse=True)
    return results, ok


def stage_best(args):
    """
    Write the best restart's parameters to JSON, to seed the next staged round.

    gather() only sees results/fit_<model>_task*.pkl, so move each round's
    pickles into a subdirectory before submitting the next one; otherwise the
    rounds pool and 'best' reports the best across all of them, which is right
    for the final answer but wrong as a seed for round N+1.
    """
    results, ok = gather(args.model)
    if not ok:
        sys.exit("no successful restarts for {} in {}/".format(args.model, OUTDIR))

    spread = ok[0]["ll"] - ok[min(9, len(ok) - 1)]["ll"]
    out = {"model": args.model, "ll": float(ok[0]["ll"]),
           "param_names": list(ok[0]["param_names"]),
           "params": [float(v) for v in ok[0]["params"]],
           "n_ok": len(ok), "n_total": len(results),
           "top10_spread": float(spread)}

    path = args.out or os.path.join(OUTDIR, "best_{}.json".format(args.model))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print("{}: {}/{} restarts ok, best ll = {:.4f}, top-10 spread = {:.3f}".format(
        args.model, len(ok), len(results), out["ll"], spread))
    for n, v in zip(out["param_names"], out["params"]):
        print("  {:8s} {:.6g}".format(n, v))

    ev = [r.get("rounds") for r in ok[:10] if r.get("rounds")]
    if ev:
        last = [d[-1] for d in ev]
        print("  last round: median {:.0f} evals, {:.0f}s "
              "(limits were maxeval and maxtime; at either one, the optimiser "
              "was cut off rather than converged)".format(
                  float(np.median([d["n_eval"] for d in last])),
                  float(np.median([d["seconds"] for d in last]))))
    print("wrote {}".format(path))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Parameter", "Value", "Description"])
        for row in rows:
            w.writerow(row)


def model_csv(model_name, best, ok, n_attempted, meta, phys, cl, rank):
    """Build the rows of one model's results CSV."""
    spec = MODELS[model_name]
    names, lb, ub = spec["names"], spec["lower"], spec["upper"]
    p = dict(zip(names, best["params"]))
    L, Nref = meta["L"], phys["Nref"]
    gt = meta["generation_time"]
    B = ["", "", ""]

    rows = [B, ["=== {}: Constants ===".format(model_name), "", ""],
            ["mu", "{:.3e}".format(meta["mu"]), "Mutation rate per bp per generation"],
            ["Generation time", "{:g}".format(gt), "Years per generation"],
            ["L", "{:d}".format(int(L)), "Callable sites"],
            ["theta", "{:.2f}".format(best["theta"]),
             "Population mutation rate (4*Nref*mu*L)"],
            B, ["=== Scaled Parameters (Coalescent Units) ===", "", ""]]
    for n in names:
        rows.append([n, "{:.4f}".format(p[n]), spec["desc"].get(n, "")])

    rows += [B, ["=== Real-World Parameters ===", "", ""],
             ["Nref", "{:.0f}".format(Nref), "Ancestral effective population size"]]
    for n, v in phys["sizes"].items():
        rows.append(["N_" + n, "{:.0f}".format(v),
                     "{} in individuals".format(spec["desc"].get(n, n))])
    for n, (gens, yrs) in phys["times"].items():
        rows.append([n + "_generations", "{:.0f}".format(gens),
                     spec["desc"].get(n, n) + ", in generations"])
        rows.append([n + "_years", "{:.0f}".format(yrs),
                     spec["desc"].get(n, n) + ", in years"])
    rows.append(["T_divergence_generations", "{:.0f}".format(phys["divergence"][1]),
                 "Total divergence time in generations ({})".format(phys["divergence_from"])])
    rows.append(["T_divergence_years", "{:.0f}".format(phys["divergence"][2]),
                 "Total divergence time in years ({})".format(phys["divergence_from"])])
    for n, v in phys["rates"].items():
        rows.append([n + "_per_gen", "{:.6e}".format(v),
                     spec["desc"].get(n, n) + ", per generation"])
    for (mig, size) in spec["migrants"]:
        if mig in phys["migrants"]:
            rows.append(["migrants_" + mig, "{:.3f}".format(phys["migrants"][mig]),
                         "Individuals per generation entering the receiving "
                         "population ({} * {} / 2)".format(size, mig)])

    rows += [B, ["=== Model Fit Statistics ===", "", ""],
             ["Log-likelihood", "{:.2f}".format(best["ll"]), "Higher is better"],
             ["Parameters (k)", "{:d}".format(len(names)),
              "Free parameters, excluding theta"]]
    if not cl:
        rows.append(["AIC", "{:.2f}".format(2 * len(names) - 2 * best["ll"]),
                     "Lower is better - UNADJUSTED, not comparable across models"])
        rows.append(["CLAIC", "not computed", "Run report with bootstraps present"])
    else:
        head_eps, head = cl[0]
        rows += [["AIC", "{:.2f}".format(head["aic"]),
                  "Lower is better - UNADJUSTED, not comparable across models"],
                 ["Effective parameters", "{:.2f}".format(head["eff_params"]),
                  "tr(J.H^-1); exceeds k+1 by however much linkage has inflated "
                  "the composite likelihood"],
                 ["CLAIC", "{:.2f}".format(head["claic"]),
                  "-2*ll + 2*tr(J.H^-1). Lower is better. THIS is the comparable one"],
                 ["dCLAIC", "{:.2f}".format(rank["delta"]),
                  "CLAIC above the best model"],
                 ["Akaike weight", "{:.3f}".format(rank["weight"]), ""],
                 ["Bootstraps", "{:d}".format(head["n_boot"]),
                  "Block bootstrap replicates behind J"],
                 ["eps", "{:g}".format(head_eps),
                  "Finite-difference step size used for the values above"]]
        for eps, res in cl:
            ev = np.linalg.eigvalsh(res["H"])
            rows.append([
                "Stability check, eps = {:g}".format(eps),
                "eff. k {:.2f}, CLAIC {:.2f}".format(res["eff_params"], res["claic"]),
                "H condition number {:.1e}, smallest eigenvalue {:+.2e}. A large "
                "move in eff. k between step sizes means the finite differences "
                "are not converged; a negative eigenvalue means the fit is not at "
                "an interior maximum and the value is meaningless."
                .format(np.linalg.cond(res["H"]), ev.min())])

    se = best.get("se_log")
    if se is not None:
        vals = list(best["params"]) + [best["theta"]]
        labels = names + ["theta"]
        rows += [B, ["=== 95% Confidence Intervals ===", "", ""]]
        for n, v, s, lo, hi in zip(labels, vals, se, list(lb) + [0.0],
                                   list(ub) + [np.inf]):
            if not np.isfinite(s):
                rows.append([n + "_CI", "not defined",
                             "Negative sandwich variance: a flat direction in H, "
                             "not an interval that happens to be wide"])
                continue
            ci_lo, ci_hi = v * np.exp(-1.96 * s), v * np.exp(1.96 * s)
            note = ("Godambe, log scale by the delta method"
                    if ci_lo >= lo and ci_hi <= hi else
                    "Godambe, log scale by the delta method. CROSSES A BOUND: "
                    "the quadratic approximation assumes an interior optimum, "
                    "so this interval is too narrow -- do not interpret the "
                    "limit that crosses")
            rows.append([n + "_CI", "{:.4g} to {:.4g}".format(ci_lo, ci_hi), note])

    lls = [r["ll"] for r in ok]
    spread = lls[0] - lls[min(9, len(lls) - 1)]
    stuck = [n for n, b in zip(names, best["at_bound"]) if b]
    rows += [B, ["=== Convergence ===", "", ""],
             ["Top-10 spread", "{:.3f}".format(spread),
              "Log-likelihood gap between best and 10th-best restart"],
             ["Converged?", "yes" if spread < 1 else "NO",
              "Above ~1 unit the optimiser has NOT converged"],
             ["Restarts succeeded", "{:d}".format(len(ok)), "Restarts returning a fit"],
             ["Restarts attempted", "{:d}".format(n_attempted), "Restarts run"],
             ["Parameters on a bound", ", ".join(stuck) if stuck else "none",
              "A fit on a bound is a constrained optimum, not a maximum"]]

    rows += [B, ["=== Caveats ===", "", ""],
             ["Why CLAIC and not AIC", "",
              "dadi maximises a composite likelihood (it treats linked sites as "
              "independent), so it overstates the information in the data and AIC's "
              "2k penalty is too small. CLAIC replaces k with tr(J.H^-1), estimated "
              "from block bootstraps. Compare models on CLAIC only."],
             ["CLAIC precision", "",
              "The trace carries sampling noise of roughly 10-15%, so CLAIC "
              "differences of a few units are not meaningful. Read only large gaps."],
             ["Reading the intervals", "",
              "These are Godambe, not Fisher: G = H.J^-1.H, so the covariance is "
              "H^-1.J.H^-1, which widens the naive interval by however much "
              "linkage has inflated the composite likelihood. They are computed "
              "on the log scale, so they are multiplicative and asymmetric about "
              "the estimate. 'not defined' means a negative sandwich variance, "
              "i.e. a flat direction in H -- that is a statement about "
              "identifiability, not a failed calculation."],
             ["Rates vs individuals", "",
              "Migrant individuals per generation is nu*M/2, which scales with the "
              "RECEIVING population, not Nref. A large rate asymmetry can come purely "
              "from a size difference, so read both numbers together."],
             ["Time parameterisation", "",
              "sec_contact's T1 and T2 are epoch DURATIONS; Dennis's TA, TB and TD are "
              "absolute times BEFORE PRESENT. Total divergence here is {}."
              .format(phys["divergence_from"])],
             ["Grid stability", "",
              "High migration strains dadi's diffusion grid. Where a rate exceeds ~10, "
              "re-evaluate at larger grid sizes and confirm the likelihood does not move."],
             ["Population order", "",
              "pop1 = {}, pop2 = {}. m12 is migration INTO pop1 FROM pop2. Column-to-"
              "population assignment was verified empirically by PCA."
              .format(*ANALYSIS_POPS)]]
    return rows


def stage_report(args):
    """
    Polish, CLAIC, CSV and figure -- one model at a time, written as it goes.

    Everything used to be written at the end, so a wall-clock kill during the
    second model's polish lost the first model's results too. Now each model's
    CSV and figure land as soon as that model is finished, and the cross-model
    comparison is a final pass that rewrites the CSVs with dCLAIC once every
    model is in. A timeout therefore costs only the model in flight.
    """
    with open(META_FILE) as f:
        meta = json.load(f)
    fs = dadi.Spectrum.from_file(FS_FILE)
    pts_l = grid_points(fs.sample_sizes)

    boot_files = sorted(glob.glob(BOOT_GLOB))
    all_boot = [dadi.Spectrum.from_file(f) for f in boot_files]
    if all_boot:
        print("{} bootstrap replicates, eps = {}".format(
            len(all_boot), ", ".join("{:g}".format(e) for e in args.eps)), flush=True)
    else:
        print("WARNING: no bootstraps found; CLAIC will be skipped", flush=True)

    done = {}
    for name in args.models:
        results, ok = gather(name)
        if not ok:
            print("no successful fits for {}, skipping".format(name), flush=True)
            continue

        best = dict(ok[0])
        if args.polish:
            t0 = time.time()
            p, ll, theta, gain = polish_fit(name, fs, best["params"],
                                            maxeval=args.maxeval,
                                            maxtime=args.polish_maxtime)
            print("{}: polished ll {:.4f} -> {:.4f} (gain {:.2e}, {:.0f}s)".format(
                name, best["ll"], ll, gain, time.time() - t0), flush=True)
            spec = MODELS[name]
            best.update({"params": np.asarray(p), "ll": ll, "theta": theta,
                         "polish_gain": gain,
                         "at_bound": [bool(np.isclose(v, lo, rtol=1e-3)
                                           or np.isclose(v, hi, rtol=1e-3))
                                      for v, lo, hi in zip(p, spec["lower"],
                                                           spec["upper"])]})

        # log=False throughout. In log space dadi's finite differences put a
        # negative eigenvalue into H on this data, which makes the trace
        # meaningless; the linear scale keeps H positive definite.
        per_eps = []
        if all_boot:
            func_ex = dadi.Numerics.make_extrap_log_func(MODELS[name]["func"])
            for eps in args.eps:
                t0 = time.time()
                res = claic.claic(func_ex, pts_l, all_boot, list(best["params"]),
                                  fs, eps=eps)
                per_eps.append((eps, res))
                print("  {} eps={:g}: CLAIC {:.2f}  eff. k {:.2f}  ({:.0f}s)".format(
                    name, eps, res["claic"], res["eff_params"], time.time() - t0),
                    flush=True)
            spread = [r["eff_params"] for _, r in per_eps]
            if len(spread) > 1 and min(spread) > 0:
                rel = (max(spread) - min(spread)) / min(spread)
                print("    eff. k varies by {:.0%} across step sizes{}".format(
                    rel, "" if rel < 0.15 else "  <- NOT CONVERGED, do not report"),
                    flush=True)

            # Confidence intervals, from the first eps only. Stored on `best`
            # rather than as a fifth element of the payload tuple, because
            # --use-cached loads claic_*.pkl written before this existed and
            # _write_model_outputs unpacks four. A new dict key means old
            # caches still load and simply skip the section.
            try:
                best["se_log"] = parameter_errors(per_eps[0][1],
                                                  best["params"], best["theta"])
                bad = int(np.sum(~np.isfinite(best["se_log"])))
                print("    intervals at eps={:g}{}".format(
                    per_eps[0][0],
                    "" if not bad else
                    "  <- {} parameter(s) with no defined interval".format(bad)),
                    flush=True)
            except Exception as e:
                print("    intervals failed ({}); CLAIC is unaffected".format(e),
                      flush=True)

        done[name] = (results, ok, best, per_eps)
        _write_model_outputs(name, done[name], meta, fs,
                             {"delta": 0.0, "weight": 1.0})
        # Cached so that a model finished in an earlier job can still appear in
        # the comparison. sec_contact takes ~5 min and basic takes hours, so
        # running them in one job just to get dCLAIC is a poor trade.
        with open(os.path.join(OUTDIR, "claic_{}.pkl".format(name)), "wb") as f:
            pickle.dump(done[name], f)

    if not done:
        sys.exit("nothing to report")

    # Pull in any other model finished in an earlier run, so the comparison is
    # complete without redoing the expensive one.
    #
    # Guarded on sample size. A cache written against a different spectrum
    # produces a dCLAIC that is arithmetically fine and scientifically empty,
    # with nothing in the output to say so -- the failure is silent, which is
    # the kind worth spending ten lines to prevent.
    this_ns = tuple(int(x) for x in fs.sample_sizes)
    if args.use_cached:
        for name in MODELS:
            cache = os.path.join(OUTDIR, "claic_{}.pkl".format(name))
            if name in done or not os.path.exists(cache):
                continue
            try:
                with open(cache, "rb") as f:
                    payload = pickle.load(f)
            except Exception as e:
                print("could not read {} ({})".format(cache, e), flush=True)
                continue
            cached_ns = payload[1][0].get("sample_sizes") if payload[1] else None
            if cached_ns is not None and tuple(cached_ns) != this_ns:
                print("REFUSING cached {}: fitted to ns={}, current spectrum is "
                      "ns={}. Log-likelihoods are not comparable across spectra. "
                      "Delete {} or re-fit the model."
                      .format(name, tuple(cached_ns), this_ns, cache), flush=True)
                continue
            done[name] = payload
            print("using cached {} (ll {:.2f}, {} restarts) from {}".format(
                name, payload[2]["ll"], len(payload[1]), cache), flush=True)

    ranks = {}
    have_claic = {n: v[3][0][1] for n, v in done.items() if v[3]}
    if have_claic:
        rows = claic.compare(have_claic)
        ranks = {r["name"]: r for r in rows}
        print("\n" + claic.format_comparison(rows), flush=True)
        with open(os.path.join(OUTDIR, "model_comparison.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "ll", "k_incl_theta", "eff_params", "CLAIC",
                        "dCLAIC", "weight", "AIC_unadjusted", "eps"])
            for r in rows:
                w.writerow([r["name"], "{:.2f}".format(r["ll"]), r["k"],
                            "{:.2f}".format(r["eff_params"]),
                            "{:.2f}".format(r["claic"]), "{:.2f}".format(r["delta"]),
                            "{:.3f}".format(r["weight"]), "{:.2f}".format(r["aic"]),
                            "{:g}".format(args.eps[0])])
        print("wrote {}".format(os.path.join(OUTDIR, "model_comparison.csv")))

    # Rewrite each CSV now that dCLAIC and the weights are known.
    if len(done) > 1 and ranks:
        for name, payload in done.items():
            _write_model_outputs(name, payload, meta, fs,
                                 ranks.get(name, {"delta": 0.0, "weight": 1.0}),
                                 plot=False)


def _write_model_outputs(name, payload, meta, fs, rank, plot=True):
    results, ok, best, per_eps = payload
    phys = rescale(name, best["params"], best["theta"], meta["L"])
    rows = model_csv(name, best, ok, len(results), meta, phys, per_eps, rank)
    path = os.path.join(OUTDIR, "{}_results.csv".format(name))
    write_csv(path, rows)
    print("wrote {}".format(path), flush=True)
    if plot:
        try:
            png = plot_fit(name, fs, best["params"], best["ll"],
                           os.path.join(OUTDIR, "fit_{}.png".format(name)))
            print("wrote {}".format(png), flush=True)
        except Exception as e:
            print("plotting failed ({}); the CSV is unaffected".format(e), flush=True)


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("sfs", help="MSMC -> data.fs + block bootstraps")
    p.add_argument("--input", nargs="+", default=INPUT_FILES)
    p.add_argument("--nboot", type=int, default=N_BOOT)
    p.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.set_defaults(func=stage_sfs)

    p = sub.add_parser("fit", help="one optimisation restart (SLURM array unit)")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--maxiter", type=int, default=300,
                   help="Nelder-Mead iterations per round (unconstrained models)")
    p.add_argument("--maxeval", type=int, default=2000,
                   help="COBYLA evaluations per round (constrained models)")
    p.add_argument("--ftol", type=float, default=1e-2,
                   help="absolute log-likelihood tolerance for COBYLA restarts; "
                        "loose on purpose, since report polishes the winner")
    p.add_argument("--maxtime", type=float, default=1800.0,
                   help="seconds per COBYLA round before it returns its best "
                        "point so far; caps a pathological restart")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--start-from", default=None,
                   help="JSON written by the 'best' stage; perturb around those "
                        "parameters instead of the model's P0")
    p.add_argument("--fold", type=float, default=1.0,
                   help="perturbation width passed to dadi.Misc.perturb_params; "
                        "shrink it round by round when staging (2, 1, 0.5, 0.25)")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=stage_fit)

    p = sub.add_parser("best", help="write the best restart to JSON, to seed "
                                    "the next staged round")
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--out", default=None,
                   help="default: results/best_<model>.json")
    p.set_defaults(func=stage_best)

    p = sub.add_parser("report", help="rescale, write CSVs, compare with CLAIC")
    p.add_argument("--models", nargs="+", default=_available(["sec_contact", "basic"]),
                   choices=list(MODELS))
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.005],
                   help="finite-difference steps for the Godambe matrices; the "
                        "first is reported, the rest are the stability check. "
                        "Smaller is NOT better: below ~1e-3 the second "
                        "differences are dominated by numerical noise")
    p.add_argument("--no-polish", dest="polish", action="store_false",
                   help="skip re-optimising the best restart before CLAIC "
                        "(not recommended; see polish_fit)")
    p.add_argument("--maxeval", type=int, default=2000,
                   help="COBYLA evaluations per polishing round")
    p.add_argument("--no-cached", dest="use_cached", action="store_false",
                   help="ignore models finished in earlier report runs instead "
                        "of folding them into the comparison")
    p.add_argument("--polish-maxtime", type=float, default=3600.0,
                   help="seconds per COBYLA polishing round before it returns "
                        "its best point so far")
    p.set_defaults(func=stage_report, polish=True, use_cached=True)

    p = sub.add_parser("plot", help="four-panel data/model/residual figure")
    p.add_argument("--models", nargs="+", default=_available(["sec_contact", "basic"]),
                   choices=list(MODELS))
    p.add_argument("--resid-range", type=float, default=None,
                   help="clip the residual colour scale at +/- this value")
    p.set_defaults(func=stage_plot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()