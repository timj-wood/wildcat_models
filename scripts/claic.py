#!/usr/bin/env python
"""
Composite-likelihood AIC (CLAIC) for comparing dadi models.

WHY NOT PLAIN AIC
-----------------
dadi's log-likelihood is a *composite* likelihood: it multiplies together
per-site contributions as though sites were independent, which they are not.
The likelihood surface is therefore too sharp, and AIC's penalty of 2k is too
small.  The composite-likelihood analogue of AIC replaces the parameter count
with an effective number of parameters:

    CLAIC = -2 * ll(p_hat) + 2 * tr(J . H^-1)

    H  sensitivity matrix, -d2/dp2 of the composite log-likelihood at the fit,
       evaluated on the real data.
    J  variability matrix, the variance of the score, estimated by averaging
       the outer product of the score over bootstrap replicates.

This is Varin & Vidoni (2005), and it is the criterion GADMA reports.  If the
model is correct and sites really are independent then J = H, the trace
collapses to k, and CLAIC reduces to ordinary AIC.  The excess of tr(J.H^-1)
over k is a direct measure of how much linkage has inflated the confidence of
the composite likelihood.

dadi does not expose CLAIC, but it computes both matrices as intermediate
results of `dadi.Godambe.get_godambe`, so nothing in dadi needs to be modified.
This module pulls them out and does the arithmetic.  It has no dependency on
the wildcat model code and can be copied into any dadi project.

YOU NEED BOOTSTRAPS
-------------------
CLAIC requires bootstrap replicates for exactly the reason `Godambe.GIM_uncert`
does: without them there is no estimate of J, and the whole correction
disappears.  They should be *block* bootstraps of the real data, built with

    chunks = dadi.Misc.fragment_data_dict(dd, chunk_size)   # several Mb chunks
    boots  = dadi.Misc.bootstraps_from_dd_chunks(chunks, Nboot, pop_ids, ns)

(see dadi's doc/user-guide/bootstrapping.md).  Blocks are used so that linkage
is preserved *within* a chunk and broken *between* chunks; that is what lets J
see the real variance of the score.  Replicates with independent sites -- for
instance `fs.sample()` Poisson draws -- give J = H by construction, the
correction cancels, and you get AIC back with extra steps.

The replicates must match the data in sample sizes, masking and folding, and
should carry the same total number of sites: the score scales with data size,
so J scales with its square, and oversized replicates inflate the penalty.

Unlike `GIM_uncert`, CLAIC inverts only H, never J, so it does not fail when
there are fewer replicates than parameters.  The trace is just noisy; treat a
handful of replicates with suspicion.

COMMAND LINE
------------
Compare two models on the same data and the same bootstraps:

    claic.py --fs data.fs --boots 'boots/*.fs' --pts 40 50 60 \\
        --model basic  --popt-json fit_basic.json \\
        --model growth --popt-json fit_growth.json

Or give parameters directly, with any dadi model by dotted path:

    claic.py --fs data.fs --boots 'boots/*.fs' --pts 40 50 60 \\
        --model dadi.Demographics2D.split_mig --popt 2.1 0.7 0.3 1.4

`--model` is repeatable and is paired in order with `--popt`/`--popt-json`.
Shorthands `basic` and `growth` refer to the wildcat models in
wildcat_models.py; anything containing a dot is imported as `module.attribute`.
Other options: `--misid` to append an ancestral-misidentification parameter,
`--fold`, `--no-multinom` if theta is an explicit parameter of your model,
`--log`, `--json out.json`, and `--eps` which accepts several step sizes.

Differences in CLAIC are only meaningful across models fitted to the *same*
data and evaluated with the *same* bootstraps, which the CLI enforces by taking
one `--fs` and one `--boots`.

FROM PYTHON
-----------
    from claic import claic, compare

    res = claic(func_ex, pts_l, boots, popt, data)
    res['claic'], res['eff_params'], res['ll'], res['warnings']

    print(compare({'basic': res_basic, 'growth': res_growth}))

CHECKING IT WORKS
-----------------
    claic.py --selftest

runs four checks that need no data; see `selftest` below for what they are and
why they are the right ones.

THREE THINGS TO WATCH
---------------------
These are properties of dadi's finite differences, not of the CLAIC formula,
but they decide whether the number you get means anything.  The figures quoted
come from `--selftest`, where the right answer is known to be 5.

* **Do not shrink eps.**  The usual advice to try smaller step sizes is wrong
  here.  `tr(J.H^-1)` is built from second derivatives of an extrapolated
  spectrum, and once the step is small enough the differences are dominated by
  numerical noise in the model rather than by curvature.  In the selftest, the
  trace drifts 5.88 -> 5.19 -> 4.07 as eps goes 0.01 -> 0.001 -> 0.0001, and at
  eps = 0.0001 dadi's own `Godambe.LRT_adjust` returns -27.3 for a quantity
  that has to be positive.  0.01 is a sensible default.  Sweep `--eps` to
  confirm stability, but read a large spread as "not converged", not as an
  invitation to go smaller.

* **Prefer the linear scale to `--log`.**  dadi's step-size rule
  (`Godambe.get_hess`) falls back to a one-sided difference whenever
  `p * eps < 1e-6`.  In log space that test is met by *every* parameter below
  1, because log(p) is then negative, and the step becomes absolute rather than
  fractional.  The trace is mathematically invariant to reparameterisation, so
  linear and log must agree, but in the selftest they differ by 29% at
  eps = 0.01 and log space returns 0.24 at eps = 0.05.  `--log` is offered for
  completeness; the linear scale is what to report.

* **The trace carries sampling noise.**  H comes from one realisation of the
  data.  With ~24,000 segregating sites the selftest's trace is 5.88 where the
  expectation is 5, and the error only falls with more data (4.93 at ten times
  the sites).  So an effective parameter count is good to roughly 10-15% at
  realistic data sizes, and CLAIC differences of a few units are not
  meaningful.  Compare models on differences much larger than that, or not at
  all.
"""
import argparse
import glob as globmod
import importlib
import json
import os
import sys

import numpy

import dadi
from dadi import Godambe, Inference


def claic(func_ex, pts, all_boot, p0, data, multinom=True, eps=0.01, log=False,
          boot_theta_adjusts=None):
    """
    Composite-likelihood AIC for a fitted dadi model.

    Args:
        func_ex (func): Model function, usually wrapped in
            `dadi.Numerics.make_extrap_func`.
        pts (list[int]): Grid points at which to evaluate func_ex.
        all_boot (list[Spectrum]): Bootstrap replicates of the data. Should be
            block bootstraps; see the module docstring.
        p0 (list[float]): Best-fit parameters.
        data (Spectrum): The observed spectrum that p0 was fitted to.
        multinom (bool): If True (the default), theta is not an explicit
            parameter of the model and is handled here, exactly as
            `Godambe.GIM_uncert` handles it. It then counts as a parameter.
        eps (float): Fractional step size for the finite differences.
        log (bool): Take derivatives with respect to log-parameters. The trace
            is invariant to this in exact arithmetic; see the module docstring
            for why the linear scale is usually more accurate.
        boot_theta_adjusts (list[float]): Per-replicate theta rescalings,
            relative to the data. Only valid with multinom=False.

    Returns:
        dict: with keys ``ll``, ``k`` (parameters counted, including theta when
        multinom), ``eff_params`` (tr(J.H^-1)), ``claic``, ``aic``, ``H``,
        ``J``, ``n_boot``, ``eps``, ``log``, and ``warnings`` (a list of
        strings; these are reported rather than raised, because a marginal
        Hessian is common and you should see the number alongside the caveat).
    """
    p0 = list(p0)
    ns = data.sample_sizes

    # Handle theta exactly as GIM_uncert does (dadi/Godambe.py, in the
    # `if multinom:` block).  get_godambe does *not* do this itself: it takes
    # func_ex and p0 at face value.  Without this, theta is neither counted in
    # the penalty nor allowed to vary when H and J are built, which biases both
    # the trace and the log-likelihood.
    if multinom:
        if boot_theta_adjusts:
            raise ValueError('boot_theta_adjusts can only be used with '
                             'multinom=False')
        func_multi = func_ex
        theta_opt = Inference.optimal_sfs_scaling(func_multi(p0, ns, pts), data)
        p0 = p0 + [theta_opt]
        func_ex = lambda p, ns, pts: p[-1] * func_multi(p[:-1], ns, pts)

    # Godambe.cache is a module-level dict keyed on func_ex.__hash__().  The
    # wrapper above is a lambda, so that hash is its id, and an id freed by the
    # garbage collector can be reused by the next model's wrapper -- which would
    # silently return another model's spectrum.  Clearing also stops the cache
    # growing without bound across a multi-model comparison.
    Godambe.cache.clear()

    k = len(p0)
    ll = float(Inference.ll(func_ex(p0, ns, pts), data))

    # eps is get_godambe's sixth positional argument.
    _, H, J, _ = Godambe.get_godambe(func_ex, pts, all_boot, p0, data, eps, log,
                                     boot_theta_adjusts=boot_theta_adjusts)

    # tr(J.H^-1) == tr(H^-1.J); solving is better conditioned than inverting.
    eff = float(numpy.trace(numpy.linalg.solve(H, J)))

    warnings = []
    eigenvalues = numpy.linalg.eigvalsh(H)
    if eigenvalues.min() <= 0:
        warnings.append(
            'H is not positive definite (smallest eigenvalue {:.3g}): p0 is '
            'not at an interior maximum, or a parameter sits on a bound.'
            .format(eigenvalues.min()))
    elif eigenvalues.max() / eigenvalues.min() > 1e10:
        warnings.append(
            'H is nearly singular (condition number {:.2g}): some parameter or '
            'combination is barely identified, so the trace is unreliable.'
            .format(eigenvalues.max() / eigenvalues.min()))
    if eff <= 0:
        warnings.append('Effective parameters {:.3g} is not positive; the '
                        'CLAIC value is meaningless.'.format(eff))
    elif eff > 5 * k:
        warnings.append('Effective parameters {:.3g} is more than five times '
                        'the {} fitted, which usually means the bootstraps do '
                        'not match the data in size or in sample sizes.'
                        .format(eff, k))
    if len(all_boot) < k:
        warnings.append('Only {} bootstrap replicates for {} parameters; J is '
                        'rank deficient and the trace is very noisy. CLAIC '
                        'still evaluates, since only H is inverted.'
                        .format(len(all_boot), k))

    return {'ll': ll, 'k': k, 'eff_params': eff,
            'claic': -2 * ll + 2 * eff, 'aic': -2 * ll + 2 * k,
            'H': H, 'J': J, 'n_boot': len(all_boot), 'eps': eps, 'log': log,
            'warnings': warnings}


def compare(results):
    """
    Rank models by CLAIC.

    Args:
        results (dict): Mapping of model name to a dict returned by `claic`.
            Every model must have been evaluated on the same data with the same
            bootstraps, or the comparison is meaningless.

    Returns:
        list[dict]: One row per model, sorted best first, each with ``name``,
        ``ll``, ``k``, ``eff_params``, ``claic``, ``delta`` (CLAIC above the
        best) and ``weight`` (the Akaike weight).
    """
    rows = [dict(name=name, ll=r['ll'], k=r['k'], eff_params=r['eff_params'],
                 claic=r['claic'], aic=r['aic'])
            for name, r in results.items()]
    rows.sort(key=lambda row: row['claic'])

    best = rows[0]['claic']
    weights = [numpy.exp(-0.5 * (row['claic'] - best)) for row in rows]
    total = sum(weights)
    for row, weight in zip(rows, weights):
        row['delta'] = row['claic'] - best
        row['weight'] = weight / total
    return rows


def format_comparison(rows):
    """Render `compare` output as a plain-text table."""
    width = max([len(row['name']) for row in rows] + [5])
    template = '{:<' + str(width) + '} {:>12} {:>4} {:>9} {:>12} {:>9} {:>8}'
    header = template.format('model', 'll', 'k', 'eff. k', 'CLAIC', 'dCLAIC',
                             'weight')
    lines = [header, '-' * len(header)]
    for row in rows:
        lines.append(template.format(
            row['name'], '{:.2f}'.format(row['ll']), row['k'],
            '{:.2f}'.format(row['eff_params']), '{:.2f}'.format(row['claic']),
            '{:.2f}'.format(row['delta']), '{:.3f}'.format(row['weight'])))
    return '\n'.join(lines)


def reduce_func(func, fixed_params):
    """
    Wrap a model so that it takes only the free parameters.

    Use this when the fit held some parameters fixed: the Hessian must then be
    taken over the free parameters only, and k must count only those. Pass the
    reduced function and the free parameters to `claic`.
    """
    if fixed_params is None:
        return func

    def reduced(params, ns, pts):
        return func(Inference._project_params_up(params, fixed_params), ns, pts)

    return reduced


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest(eps_list=(0.01,), n_boot=200, seed=20260806, verbose=True):
    """
    Check the calculation against four things that must be true.

    Uses `dadi.Demographics2D.split_mig` rather than any wildcat model, so this
    file stands on its own, and evaluates at the simulating parameters, which is
    where the information equality holds exactly and where no optimisation is
    needed.

    1. Information equality. With replicates that really are independent, J = H
       and tr(J.H^-1) must equal the number of parameters. This is the check
       that catches a mishandled theta, which is the easiest thing to get wrong.
       It is applied with H taken from the noiseless expected spectrum, so that
       it tests the arithmetic rather than the luck of one realisation; the
       value from the simulated data is reported alongside to show how much
       sampling noise the trace carries in practice.
    2. Information equality, at the matrix level. The reason check 1 works is
       that J and H both estimate the Fisher information, so they must agree
       entrywise and not merely in trace, which could come right by
       cancellation. The residual is Monte Carlo error in J and falls off as
       1/sqrt(n_boot): about 0.15 at 200 replicates, 0.13 at 500.
    3. Response to linkage, against an exact answer. If every site is observed
       twice -- perfect pairwise linkage -- the composite log-likelihood is
       exactly doubled (up to a parameter-free constant), so the score doubles,
       H doubles, J quadruples, and tr(J.H^-1) must double. Doubling the data
       and the replicates must therefore double the effective parameter count,
       and CLAIC's penalty must rise while AIC's does not. The factor of two
       here is algebraically exact rather than approached numerically, so this
       check confirms the response to linkage and the consistency of the theta
       handling under rescaling; it is not an independent numerical test.
    4. Agreement with dadi. `Godambe.LRT_adjust` returns
       len(nested)/tr(J.H^-1) over a nested subspace, so it must reproduce our
       trace on the same subspace. An independent check against dadi's own code.

    Two further numbers are printed without a pass/fail, because they measure
    the accuracy of dadi's finite differences rather than anything this module
    does: the trace with H taken from the noisy simulated data, and the trace
    computed in log space. See the module docstring on both.

    Returns:
        bool: True if every check passed.
    """
    func = dadi.Demographics2D.split_mig
    func_ex = dadi.Numerics.make_extrap_func(func)
    p_true = [2.0, 0.5, 0.3, 1.0]
    ns, pts_l, theta = (12, 12), [30, 40, 50], 5000.0
    k_expected = len(p_true) + 1          # + theta, since multinom=True

    numpy.random.seed(seed)
    expected = func_ex(p_true, ns, pts_l) * theta
    data = expected.sample()

    if verbose:
        print('Model {}, p = {}, theta = {:.0f}, ns = {}, pts = {}'
              .format(func.__name__, p_true, theta, ns, pts_l))
        print('Simulated data: {:.0f} segregating sites\n'.format(data.S()))

    unlinked = [expected.sample() for _ in range(n_boot)]
    # Every site seen twice, i.e. perfectly linked pairs.
    doubled = [2 * b for b in unlinked]

    passed = True
    for eps in eps_list:
        if verbose:
            print('eps = {:g}'.format(eps))

        exact = claic(func_ex, pts_l, unlinked, p_true, expected, eps=eps)
        linear = claic(func_ex, pts_l, unlinked, p_true, data, eps=eps)
        ok = abs(exact['eff_params'] - k_expected) / k_expected < 0.05
        passed &= _report('information equality: unlinked replicates give '
                          'eff. k = {:.3f}, expected {}'
                          .format(exact['eff_params'], k_expected), ok, verbose)

        residual = (numpy.linalg.norm(exact['J'] - exact['H'])
                    / numpy.linalg.norm(exact['H']))
        ok = residual < 0.25
        passed &= _report('information equality entrywise, not just in trace: '
                          '||J-H||/||H|| = {:.3f} over {} replicates'
                          .format(residual, n_boot), ok, verbose)

        linked = claic(func_ex, pts_l, doubled, p_true, 2 * data, eps=eps)
        ratio = linked['eff_params'] / linear['eff_params']
        ok = abs(ratio - 2.0) < 0.02
        passed &= _report('perfectly linked pairs double the effective '
                          'parameter count: {:.3f} vs {:.3f}, ratio {:.4f}, '
                          'expected exactly 2'
                          .format(linked['eff_params'], linear['eff_params'],
                                  ratio), ok, verbose)

        ours, theirs = _lrt_adjust_crosscheck(func_ex, pts_l, unlinked, p_true,
                                              data, [3], eps)
        ok = abs(ours - theirs) / abs(theirs) < 1e-6
        passed &= _report('agrees with dadi.Godambe.LRT_adjust: {:.10f} vs '
                          '{:.10f}'.format(ours, theirs), ok, verbose)

        if verbose:
            logged = claic(func_ex, pts_l, unlinked, p_true, expected, eps=eps,
                           log=True)
            print('   (no pass/fail, these measure dadi\'s finite differences: '
                  'trace is {:.3f} with H from the noisy simulated data rather '
                  'than its expectation, and {:.3f} in log space, where it '
                  'should be identical to {:.3f} because the trace is '
                  'reparameterisation invariant)'
                  .format(linear['eff_params'], logged['eff_params'],
                          exact['eff_params']))
            print('    unlinked: CLAIC {:.2f} vs AIC {:.2f}. Doubled data: '
                  'CLAIC penalty {:.2f} against AIC\'s unchanged {:.2f}, which '
                  'is the correction doing its job.\n'
                  .format(linear['claic'], linear['aic'],
                          2 * linked['eff_params'], 2 * linked['k']))

    return bool(passed)


def _report(message, ok, verbose):
    if verbose:
        print('  [{}] {}'.format('pass' if ok else 'FAIL', message))
    return ok


def _lrt_adjust_crosscheck(func_ex, pts, all_boot, p0, data, nested_indices,
                           eps):
    """
    Compute LRT_adjust two ways: through this module, and through dadi.

    Mirrors the internals of `Godambe.LRT_adjust` -- append theta, restrict to
    the nested subspace -- then takes the trace with `claic` instead of dadi's
    own code. The two must agree, since the adjustment is
    len(nested)/tr(J.H^-1).
    """
    ns = data.sample_sizes
    theta_opt = Inference.optimal_sfs_scaling(func_ex(p0, ns, pts), data)
    p_ext = numpy.asarray(list(p0) + [theta_opt], dtype=float)
    scaled = lambda p, ns, pts: p[-1] * func_ex(p[:-1], ns, pts)

    def diff_func(diff_params, ns, pts):
        full = numpy.array(p_ext, copy=True)
        full[nested_indices] = diff_params
        return scaled(full, ns, pts)

    # multinom=False: theta is already an explicit parameter of `scaled`.
    res = claic(diff_func, pts, all_boot, p_ext[nested_indices], data,
                multinom=False, eps=eps)
    ours = len(nested_indices) / res['eff_params']

    Godambe.cache.clear()
    theirs = Godambe.LRT_adjust(func_ex, pts, all_boot, p0, data,
                                nested_indices, multinom=True, eps=eps)
    return ours, float(theirs)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def resolve_model(spec):
    """Turn a --model argument into a model function."""
    if '.' not in spec:
        try:
            import wildcat_models
        except ImportError:
            raise ValueError(
                'Model {!r} is not a dotted path, and wildcat_models could not '
                'be imported to look it up. Use e.g. '
                'dadi.Demographics2D.split_mig.'.format(spec))
        if spec not in wildcat_models.MODELS:
            raise ValueError('Unknown model {!r}; wildcat_models offers {}.'
                             .format(spec, sorted(wildcat_models.MODELS)))
        return wildcat_models.MODELS[spec]

    module_name, _, attribute = spec.rpartition('.')
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise ValueError('Could not import {!r} from {!r}.'
                         .format(attribute, module_name))
    if not hasattr(module, attribute):
        raise ValueError('{!r} has no attribute {!r}.'
                         .format(module_name, attribute))
    return getattr(module, attribute)


def load_popt(spec):
    """
    Read best-fit parameters from a JSON file.

    Accepts either ``{"popt": [...]}`` or the shape written by
    demo_wildcat.py, ``{"fits": [{"popt": [...], "ll": ...}, ...]}``, in which
    case the fit with the highest log-likelihood is used.
    """
    with open(spec) as fid:
        contents = json.load(fid)
    if isinstance(contents, list):
        return [float(v) for v in contents]
    if 'popt' in contents:
        return [float(v) for v in contents['popt']]
    if 'fits' in contents and contents['fits']:
        best = max(contents['fits'], key=lambda fit: fit['ll'])
        return [float(v) for v in best['popt']]
    raise ValueError('{}: expected a list, a "popt" key, or a non-empty '
                     '"fits" list.'.format(spec))


class _CollectPopt(argparse.Action):
    """
    Gather --popt and --popt-json into one list, in command-line order.

    argparse keeps a separate list per option, which loses the interleaving and
    would pair the parameters with the wrong --model as soon as the two forms
    are mixed.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        popts = getattr(namespace, 'popts', None) or []
        if option_string == '--popt-json':
            popts.append(load_popt(values))
        else:
            popts.append([float(v) for v in values])
        namespace.popts = popts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Composite-likelihood AIC for dadi models.',
        epilog='See the module docstring for what CLAIC is and what the '
               'bootstraps have to be.')
    parser.add_argument('--selftest', action='store_true',
                        help='run the built-in checks and exit; needs no data')
    parser.add_argument('--fs', help='data spectrum, in dadi .fs format')
    parser.add_argument('--boots', action='append', default=[],
                        help='glob for bootstrap spectra, e.g. "boots/*.fs"; '
                             'may be repeated')
    parser.add_argument('--model', action='append', default=[],
                        help='"basic", "growth", or a dotted path such as '
                             'dadi.Demographics2D.split_mig; repeat to compare')
    parser.add_argument('--popt', action=_CollectPopt, dest='popts', nargs='+',
                        default=None, help='best-fit parameters for the '
                                           'corresponding --model')
    parser.add_argument('--popt-json', action=_CollectPopt, dest='popts',
                        default=None,
                        help='read the parameters from a JSON file instead; '
                             'may be interleaved with --popt')
    parser.add_argument('--pts', type=int, nargs=3, default=None,
                        help='grid points; default is ns+20, +30, +40')
    parser.add_argument('--eps', type=float, nargs='+', default=[0.01],
                        help='finite-difference step sizes; give several to '
                             'check the trace is stable')
    parser.add_argument('--misid', action='store_true',
                        help='append an ancestral misidentification parameter')
    parser.add_argument('--fold', action='store_true',
                        help='fold the data and the bootstraps')
    parser.add_argument('--no-multinom', action='store_true',
                        help='theta is an explicit parameter of the model')
    parser.add_argument('--log', action='store_true',
                        help='take derivatives in log-parameters')
    parser.add_argument('--json', help='write the full results here')
    args = parser.parse_args(argv)

    if args.selftest:
        return 0 if selftest(eps_list=args.eps) else 1

    if not args.fs or not args.model:
        parser.error('--fs and at least one --model are required '
                     '(or use --selftest)')
    if not args.boots:
        parser.error('--boots is required: without bootstrap replicates there '
                     'is no estimate of J and no correction to make. See the '
                     'module docstring.')

    popts = args.popts or []
    if len(popts) != len(args.model):
        parser.error('got {} model(s) but {} parameter set(s); each --model '
                     'needs one --popt or --popt-json'
                     .format(len(args.model), len(popts)))

    data = dadi.Spectrum.from_file(args.fs)
    boot_files = sorted(f for pattern in args.boots
                        for f in globmod.glob(pattern))
    if not boot_files:
        parser.error('no files matched {}; quote the pattern so the shell does '
                     'not expand it'.format(args.boots))
    all_boot = [dadi.Spectrum.from_file(f) for f in boot_files]
    if args.fold:
        data = data.fold()
        all_boot = [b.fold() for b in all_boot]

    ns = [int(n) for n in data.sample_sizes]
    pts = args.pts if args.pts else [max(ns) + 20, max(ns) + 30, max(ns) + 40]

    if args.log:
        print('warning: --log uses dadi\'s log-scale finite differences, which '
              'are noticeably less accurate than the linear ones. See the '
              'module docstring.\n')
    if min(args.eps) < 1e-3:
        print('warning: step sizes below 1e-3 make the second differences '
              'numerically unstable, and smaller is not better here. See the '
              'module docstring.\n')

    print('Data {}: ns = {}, {:.0f} segregating sites'
          .format(os.path.basename(args.fs), ns, data.S()))
    print('{} bootstrap replicates, grid points {}\n'
          .format(len(all_boot), pts))

    output = {'fs': args.fs, 'n_boot': len(all_boot), 'pts': pts,
              'multinom': not args.no_multinom, 'log': args.log, 'models': {}}

    # The same model may legitimately appear twice, compared at two different
    # fits, so labels have to be made unique before they become dict keys.
    labels = [spec if args.model.count(spec) == 1
              else '{}#{}'.format(spec, i + 1)
              for i, spec in enumerate(args.model)]

    for eps in args.eps:
        results = {}
        for label, spec, popt in zip(labels, args.model, popts):
            try:
                func = resolve_model(spec)
            except ValueError as err:
                parser.error(str(err))
            if args.misid:
                func = dadi.Numerics.make_anc_state_misid_func(func)
            func_ex = dadi.Numerics.make_extrap_func(func)
            res = claic(func_ex, pts, all_boot, popt, data,
                        multinom=not args.no_multinom, eps=eps, log=args.log)
            results[label] = res
            entry = output['models'].setdefault(label, {'model': spec})
            entry['popt'] = popt
            entry.setdefault('by_eps', {})[str(eps)] = {
                key: res[key] for key in
                ('ll', 'k', 'eff_params', 'claic', 'aic', 'warnings')}

        print('eps = {:g}'.format(eps))
        print(format_comparison(compare(results)))
        for label, res in results.items():
            for warning in res['warnings']:
                print('  warning [{}]: {}'.format(label, warning))
        print()

    if len(args.eps) > 1:
        print('If eff. k moves materially between step sizes, the finite '
              'differences are not converged and the CLAIC values should not '
              'be trusted.')

    if args.json:
        with open(args.json, 'w') as fid:
            json.dump(output, fid, indent=2)
        print('Full results written to {}'.format(args.json))
    return 0


if __name__ == '__main__':
    sys.exit(main())
