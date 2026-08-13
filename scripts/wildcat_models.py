"""
Two-population demographic models for Scottish wildcat and domestic cat.

These implement the model drawn in ``scottishwildcats.pdf`` (M. Beaumont), with
background from Howard-McCombe et al. (2021) Mol Ecol 30:3688-3702.  See
``wildcat_model_summary.pdf`` for a prose description and for the interpretation
choices made here.

Population 1 is the Scottish wildcat (the *Felis silvestris* lineage), population
2 is the domestic cat (the *F. lybica* lineage).  The reference population is the
common ancestor, so all sizes are relative to ``NA``, times are in units of
``2*NA`` generations, and migration rates are ``M = 2*NA*m`` where ``m`` is the
per-generation probability that a lineage traces back into the other population.

dadi's convention is that ``m12`` is the rate *into pop 1 from pop 2*, equal to
the fraction of pop 1 that are new migrants from pop 2 each generation.  Read
backwards in time that is the probability a lineage sampled in pop 1 came from
pop 2, which is exactly what the slide's ``m`` symbols denote.  So the slide's
``m_ls`` (into silvestris from lybica) is dadi's ``m12``, and ``m_sl`` is ``m21``.

Timeline, in absolute time before the present:

    TA                       split of silvestris and lybica from the ancestor
    max(TB, TD)              first of the two size changes
    min(TB, TD)              second of the two size changes
    0                        present

The slide gives TB (British landbridge break, wildcat size change) and TD
(domestication, domestic size change) as distinct times, but dadi allows only one
migration matrix per integration epoch, so the switch from (m_ls, m_sl) to
(m2_ds, m2_sd) cannot happen separately for each population.  Here the second
migration regime starts at min(TB, TD), i.e. only once *both* "Scottish wildcat"
and "Domestic" exist; the intervening epoch keeps the early rates.  To switch at
max(TB, TD) instead, move the ``m12``/``m21`` arguments of the middle
``Integration.two_pops`` call to the second-epoch rates.

Two models are defined: ``wildcat_domestic`` (instantaneous size changes, 11
parameters) and ``wildcat_domestic_growth`` (exponential size changes, 13).  A
duration-parameterised variant was contemplated at one point and its bounds were
briefly present in the tables below without a matching function; both are gone.
Any model added here must have a function, a ``__param_names__``, and one entry
each in P0, LOWER_BOUNDS and UPPER_BOUNDS, all of the same length.
"""
from dadi import Integration, Numerics, PhiManip
from dadi.Spectrum_mod import Spectrum

#: Population labels attached to spectra produced by these models.
POP_IDS = ['wildcat', 'domestic']


def wildcat_domestic(params, ns, pts):
    """
    Basic Scottish wildcat / domestic cat model, with instantaneous size changes.

    Parameters:
        params (tuple): (nuS, nuL, nuB, nuD, TA, TB, TD, m_ls, m_sl, m2_ds, m2_sd)

            - nuS: Size of the silvestris branch after the split, relative to NA.

            - nuL: Size of the lybica branch after the split, relative to NA.

            - nuB: Size of the Scottish wildcat from TB to the present.

            - nuD: Size of the domestic cat from TD to the present.

            - TA: Time before present of the silvestris/lybica split, in units
              of 2*NA generations.  Must exceed both TB and TD.

            - TB: Time before present of the wildcat size change (breaking of the
              British landbridge).

            - TD: Time before present of the domestic size change (domestication).

            - m_ls: Early gene flow into silvestris from lybica (2*NA*m).

            - m_sl: Early gene flow into lybica from silvestris (2*NA*m).

            - m2_ds: Recent gene flow into the wildcat from the domestic (2*NA*m).

            - m2_sd: Recent gene flow into the domestic from the wildcat (2*NA*m).
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    nuS, nuL, nuB, nuD, TA, TB, TD, m_ls, m_sl, m2_ds, m2_sd = params

    T_first, T_second = max(TB, TD), min(TB, TD)
    # Durations of the three post-split epochs.  Clipping at zero keeps an
    # out-of-order proposal from raising inside an optimizer; such proposals are
    # excluded properly by time_ordering_constraint.
    T1 = max(TA - T_first, 0)
    Tmid = T_first - T_second
    T2 = T_second

    # In the middle epoch, only the population whose event came first has changed.
    if TB > TD:
        nu1_mid, nu2_mid = nuB, nuL
    else:
        nu1_mid, nu2_mid = nuS, nuD

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, deme_ids=['ancestral'])
    phi = PhiManip.phi_1D_to_2D(xx, phi, deme_ids=POP_IDS)

    phi = Integration.two_pops(phi, xx, T1, nuS, nuL,
                               m12=m_ls, m21=m_sl, deme_ids=POP_IDS)
    phi = Integration.two_pops(phi, xx, Tmid, nu1_mid, nu2_mid,
                               m12=m_ls, m21=m_sl, deme_ids=POP_IDS)
    phi = Integration.two_pops(phi, xx, T2, nuB, nuD,
                               m12=m2_ds, m21=m2_sd, deme_ids=POP_IDS)

    return Spectrum.from_phi(phi, ns, (xx, xx), pop_ids=POP_IDS)


wildcat_domestic.__param_names__ = ['nuS', 'nuL', 'nuB', 'nuD', 'TA', 'TB', 'TD',
                                    'm_ls', 'm_sl', 'm2_ds', 'm2_sd']


def wildcat_domestic_growth(params, ns, pts):
    """
    Scottish wildcat / domestic cat model with exponential size change.

    This is the "tweak" suggested on the slide: instead of constant sizes with
    instantaneous jumps, each branch changes exponentially from its post-split
    size to its post-event size, and then again from its post-event size to a
    present-day size.

    Parameters:
        params (tuple): (nuS, nuL, nuB, nuD, nuBcurr, nuDcurr,
        TA, TB, TD, m_ls, m_sl, m2_ds, m2_sd)

            - nuS: Size of the silvestris branch immediately after the split.

            - nuL: Size of the lybica branch immediately after the split.

            - nuB: Size of the wildcat at TB, reached exponentially from nuS.

            - nuD: Size of the domestic at TD, reached exponentially from nuL.

            - nuBcurr: Present-day wildcat size, reached exponentially from nuB.

            - nuDcurr: Present-day domestic size, reached exponentially from nuD.

            - TA, TB, TD: Event times before present, in units of 2*NA
              generations, as in `wildcat_domestic`.

            - m_ls, m_sl, m2_ds, m2_sd: Migration rates, as in `wildcat_domestic`.
        ns (tuple): Sample sizes (n1, n2).
        pts (int): Number of grid points to use in integration.

    Returns:
        fs (Spectrum): The resulting frequency spectrum.
    """
    (nuS, nuL, nuB, nuD, nuBcurr, nuDcurr,
     TA, TB, TD, m_ls, m_sl, m2_ds, m2_sd) = params

    T_first, T_second = max(TB, TD), min(TB, TD)
    T1 = max(TA - T_first, 0)      # clipped, as in wildcat_domestic
    Tmid = T_first - T_second
    T2 = T_second

    # Size histories are written as functions of tau, the time *since* the split,
    # so that they are independent of where the epoch boundaries happen to fall.
    nu1_of_tau = _two_phase_exponential(nuS, nuB, nuBcurr, TA - TB, TB)
    nu2_of_tau = _two_phase_exponential(nuL, nuD, nuDcurr, TA - TD, TD)

    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx, deme_ids=['ancestral'])
    phi = PhiManip.phi_1D_to_2D(xx, phi, deme_ids=POP_IDS)

    tau0 = 0.0
    for T, m12, m21 in [(T1, m_ls, m_sl), (Tmid, m_ls, m_sl), (T2, m2_ds, m2_sd)]:
        phi = Integration.two_pops(
            phi, xx, T,
            nu1=_shift(nu1_of_tau, tau0), nu2=_shift(nu2_of_tau, tau0),
            m12=m12, m21=m21, deme_ids=POP_IDS)
        tau0 += T

    return Spectrum.from_phi(phi, ns, (xx, xx), pop_ids=POP_IDS)


wildcat_domestic_growth.__param_names__ = [
    'nuS', 'nuL', 'nuB', 'nuD', 'nuBcurr', 'nuDcurr',
    'TA', 'TB', 'TD', 'm_ls', 'm_sl', 'm2_ds', 'm2_sd']


def _two_phase_exponential(nu_start, nu_mid, nu_end, T_early, T_late):
    """
    Size as a function of tau, time elapsed since the split.

    Grows exponentially from nu_start to nu_mid over the first T_early, then from
    nu_mid to nu_end over the following T_late.
    """
    def nu(tau):
        if tau <= T_early:
            if T_early <= 0:
                return nu_mid
            return nu_start * (nu_mid / nu_start) ** (tau / T_early)
        if T_late <= 0:
            return nu_end
        frac = min((tau - T_early) / T_late, 1.0)
        return nu_mid * (nu_end / nu_mid) ** frac
    return nu


def _shift(func, offset):
    """Re-express a function of absolute tau as a function of an epoch-local time."""
    return lambda t: func(offset + t)


def make_time_ordering_constraint(func, fixed_params=None):
    """
    Build the inequality constraint enforcing TA > max(TB, TD) for a model.

    The returned function gives a quantity that must be <= 0, in the form
    expected by the ``ineq_constraints`` argument of `dadi.Inference.opt`.  Use
    it with a constraint-capable algorithm such as ``nlopt.LN_COBYLA``::

        cons = make_time_ordering_constraint(wildcat_domestic)
        dadi.Inference.opt(p0, data, func_ex, pts_l,
                           ineq_constraints=[(cons, 1e-6)],
                           algorithm=nlopt.LN_COBYLA, ...)

    Pass the same ``fixed_params`` you pass to `dadi.Inference.opt`.  dadi hands
    constraints straight to nlopt, which calls them with the *reduced* parameter
    vector that fixed parameters have been projected out of, so a constraint
    written against full-length indices silently constrains the wrong
    parameters.  The constraint below undoes that projection first.

    Indices come from the model's ``__param_names__``, so the constraint also
    survives wrappers that append parameters to the end, such as
    `dadi.Numerics.make_anc_state_misid_func`.
    """
    names = func.__param_names__
    iTA, iTB, iTD = names.index('TA'), names.index('TB'), names.index('TD')

    def time_ordering_constraint(params, grad=None):
        p = _project_up(params, fixed_params)
        return max(p[iTB], p[iTD]) - p[iTA]

    return time_ordering_constraint


def _project_up(params, fixed_params):
    """
    Reinsert fixed values into a reduced parameter vector.

    Mirrors `dadi.Inference._project_params_down`, which is what removed them.
    """
    if fixed_params is None:
        return params
    full, free = [], iter(params)
    for value in fixed_params:
        full.append(next(free) if value is None else value)
    return full


# Starting values and bounds.  The defaults below are rough, in the sense that
# they are meant to put an optimizer in a sensible region rather than to encode
# strong prior beliefs; see units.py for translating them to and from cats and
# years.  They assume NA is of order 1e5, so that a 1.4 Myr split at a 3 year
# generation time is TA ~ 2, and the Holocene events are TB, TD ~ 1e-2.
P0 = {
    'wildcat_domestic': [0.5, 2.0, 0.05, 0.5, 2.0, 0.02, 0.02,
                         0.2, 0.2, 0.5, 0.2],
    'wildcat_domestic_growth': [0.5, 2.0, 0.05, 0.5, 0.05, 1.0, 2.0, 0.02, 0.02,
                                0.2, 0.2, 0.5, 0.2],
}

LOWER_BOUNDS = {
    'wildcat_domestic': [1e-3, 1e-3, 1e-4, 1e-3, 1e-2, 1e-4, 1e-4,
                         0, 0, 0, 0],
    'wildcat_domestic_growth': [1e-3, 1e-3, 1e-4, 1e-3, 1e-4, 1e-3,
                                1e-2, 1e-4, 1e-4, 0, 0, 0, 0],
}

UPPER_BOUNDS = {
    'wildcat_domestic': [50, 50, 10, 50, 100, 5, 5,
                         20, 20, 20, 20],
    'wildcat_domestic_growth': [50, 50, 10, 50, 10, 50,
                                100, 5, 5, 20, 20, 20, 20],
}

#: Model functions by short name, for command-line selection.
MODELS = {
    'basic': wildcat_domestic,
    'growth': wildcat_domestic_growth,
}


def model_defaults(func):
    """Return (param_names, p0, lower_bound, upper_bound) for a model function."""
    key = func.__name__
    return (list(func.__param_names__), list(P0[key]),
            list(LOWER_BOUNDS[key]), list(UPPER_BOUNDS[key]))