"""Correlation heatmap from the Godambe matrices already computed for CLAIC.

Usage:  python plot_corr.py [model ...]     (default: all three)

Reads results_wild/claic_<model>.pkl, written by `wildcat_pipeline.py report`.
Computes nothing: H and J come from the pickle, so this is the same sandwich
covariance, H^-1 J H^-1, whose diagonal produced the published intervals.

Writes results_wild/corr_<model>.pdf and results_wild/corr_<model>.csv, and
prints three checks: se/value against the stored errors, the strongest pairs,
and the largest correlation each parameter has with any other.

Tick labels follow the report's notation (N-prefix for basic and growth, nu for
sec_contact); the printed diagnostics keep the parameter names used in the code.
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wildcat_pipeline as wp

names_wanted = sys.argv[1:] or ["basic", "growth", "sec_contact"]

# Order follows wildcat_models.py, NOT the order the parameters appear in the
# report's tables: growth puts nuBcurr and nuDcurr fifth and sixth.
PRETTY = {
    "basic": [r"$N_S$", r"$N_L$", r"$N_B$", r"$N_D$", r"$T_A$", r"$T_B$",
              r"$T_D$", r"$m_{ls}$", r"$m_{sl}$", r"$m_{2,ds}$",
              r"$m_{2,sd}$", r"$\theta$"],
    "growth": [r"$N_S$", r"$N_L$", r"$N_B$", r"$N_D$", r"$N_{B\mathrm{curr}}$",
               r"$N_{D\mathrm{curr}}$", r"$T_A$", r"$T_B$", r"$T_D$",
               r"$m_{ls}$", r"$m_{sl}$", r"$m_{2,ds}$", r"$m_{2,sd}$",
               r"$\theta$"],
    "sec_contact": [r"$\nu_1$", r"$\nu_2$", r"$m_{12}$", r"$m_{21}$",
                    r"$T_1$", r"$T_2$", r"$\theta$"],
}

for NAME in names_wanted:
    path = os.path.join(wp.OUTDIR, "claic_{}.pkl".format(NAME))
    if not os.path.exists(path):
        print("no {}, skipping".format(path), flush=True)
        continue
    with open(path, "rb") as f:
        results, ok, best, per_eps = pickle.load(f)
    if not per_eps:
        print("{}: no Godambe matrices in the pickle, skipping".format(NAME))
        continue

    eps, res = per_eps[0]
    H, J = np.asarray(res["H"]), np.asarray(res["J"])
    Hinv = np.linalg.inv(H)
    C = Hinv @ J @ Hinv                       # as in parameter_errors

    var = np.diag(C)
    se = np.sqrt(np.where(var > 0, var, np.nan))
    R = C / np.outer(se, se)

    labels = list(wp.MODELS[NAME]["names"]) + ["theta"]
    assert len(labels) == H.shape[0], (NAME, len(labels), H.shape)

    ticks = PRETTY.get(NAME, labels)
    assert len(ticks) == len(labels), (NAME, len(ticks), len(labels))

    # se/value must reproduce the stored errors behind the published intervals
    vals = np.append(np.asarray(best["params"], float), best["theta"])
    print("\n=== {} (eps = {:g})      mine     stored".format(NAME, eps))
    for lab, mine, stored in zip(labels, se / vals,
                                 best.get("se_log", se * np.nan)):
        print("{:>9}  {:.6f}  {:.6f}".format(lab, mine, stored))

    iu = np.triu_indices(len(labels), 1)
    order = np.argsort(-np.abs(R[iu]))
    print("  strongest pairs:")
    for idx in order[:12]:
        i, j = iu[0][idx], iu[1][idx]
        print("    {:>8} {:>8}  {:+.3f}".format(labels[i], labels[j], R[i, j]))

    # Backs the claims that a parameter is NOT strongly correlated with
    # anything, which the top-twelve list can only support by omission.
    off = np.abs(R - np.diag(np.diag(R)))
    print("  max |r| per parameter:")
    for lab, m, k in zip(labels, np.nanmax(off, axis=1), np.nanargmax(off, axis=1)):
        print("    {:>8}  {:.3f}  (with {})".format(lab, m, labels[k]))

    np.savetxt(os.path.join(wp.OUTDIR, "corr_{}.csv".format(NAME)),
               R, delimiter=",", fmt="%.3f",
               header=",".join(labels), comments="")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticks(range(len(ticks))); ax.set_yticklabels(ticks)
    fig.colorbar(im, label="correlation")
    fig.tight_layout()
    out = os.path.join(wp.OUTDIR, "corr_{}.pdf".format(NAME))
    fig.savefig(out)
    plt.close(fig)
    print("wrote {} and {}".format(out, out[:-4] + ".csv"), flush=True)