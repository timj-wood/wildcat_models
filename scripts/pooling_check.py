#!/usr/bin/env python
"""
Reproduce the numbers in Section 2.2 of the report: Table 2 (effect of
pooling the captive Scottish cats with the wild-caught ones) and the
nucleotide-diversity comparison between the two Scottish groups.

Run from the scripts/ directory:

    python pooling_check.py

It reads the two multihetsep files in ../data via parse_msmc() from
wildcat_pipeline.py and builds four folded joint spectra against the
domestic sample:

    wild-caught only     (32, 12)   the spectrum used for all fits
    pooled, full size    (52, 12)   wild-caught + captive
    pooled, projected    (32, 12)   the (52, 12) spectrum projected down
    captive only         (20, 12)

Nothing here is used by the fitting pipeline; it exists so that every
figure quoted in Section 2.2 can be regenerated.
"""

import sys
import numpy as np
import dadi

sys.path.insert(0, ".")
import wildcat_pipeline as wp

INPUTS = ["../data/msmc_input_chr1.txt", "../data/msmc_input_chr2.txt"]
INTERMEDIATE_BINS = range(8, 17)   # bins 8-16 of the folded 33-bin Scottish marginal

# parse_msmc() checks that every genotype string has as many characters as
# there are haplotypes across POPS, so the four groups must partition 0-91.
wp.POPS.clear()
wp.POPS.update({
    "SCO_WILD": list(range(0, 32)),
    "SCO_CAPT": list(range(32, 52)),
    "DOM":      list(range(52, 64)),
    "EUR":      list(range(64, 92)),
})

dd, L, stats = wp.parse_msmc(INPUTS)
print("\n{:,} biallelic sites, L = {:,}\n".format(stats["n_sites"], L))

# Add a pooled Scottish population to every site's call record.
for site in dd.values():
    w, c = site["calls"]["SCO_WILD"], site["calls"]["SCO_CAPT"]
    site["calls"]["SCO_POOL"] = (w[0] + c[0], w[1] + c[1])


def build(pop, n):
    return dadi.Spectrum.from_data_dict(dd, [pop, "DOM"], [n, 12], polarized=False)


def hudson_fst(fs):
    """
    Hudson's FST (Hudson et al. 1992) from a 2D spectrum, as the ratio of
    averages 1 - mean(Hw) / mean(Hb) recommended by Bhatia et al. (2013).
    Hw is the mean within-population heterozygosity and Hb the
    between-population heterozygosity, summed over unmasked entries. Every
    term is invariant under (i, j) -> (n1 - i, n2 - j), so folding does not
    affect the result. dadi's own Spectrum.Fst() is Weir-Cockerham and
    gives a different number.
    """
    n1, n2 = (int(n) for n in fs.sample_sizes)
    i = np.arange(n1 + 1)[:, None]
    j = np.arange(n2 + 1)[None, :]
    hw = (i * (n1 - i) / (n1 * (n1 - 1)) + j * (n2 - j) / (n2 * (n2 - 1)))
    hb = (i * (n2 - j) + j * (n1 - i)) / (n1 * n2)
    counts = np.ma.filled(fs, 0.0)
    return float(1 - (counts * hw).sum() / (counts * hb).sum())


def intermediate_mass(fs):
    """Share of the folded Scottish marginal spectrum in bins 8-16."""
    marg = fs.marginalize([1])
    return float(sum(marg[i] for i in INTERMEDIATE_BINS) / marg.S())


spectra = {
    "wild-caught (32,12)":      build("SCO_WILD", 32),
    "pooled full (52,12)":      build("SCO_POOL", 52),
    "pooled projected (32,12)": build("SCO_POOL", 32),
    "captive (20,12)":          build("SCO_CAPT", 20),
}

print("{:<26} {:>12} {:>10} {:>11} {:>10} {:>12}".format(
    "spectrum", "S", "Tajima D", "Hudson FST", "mass 8-16", "pi (Sco)"))
for name, fs in spectra.items():
    sco = fs.marginalize([1])
    mass = intermediate_mass(fs) if fs.sample_sizes[0] == 32 else float("nan")
    print("{:<26} {:>12,.1f} {:>10.3f} {:>11.3f} {:>10.3f} {:>12.3e}".format(
        name, fs.S(), sco.Tajima_D(), hudson_fst(fs), mass, sco.pi() / L))

pi_wild = spectra["wild-caught (32,12)"].marginalize([1]).pi() / L
pi_capt = spectra["captive (20,12)"].marginalize([1]).pi() / L
print("\ncaptive pi is {:.0%} below wild-caught pi".format(1 - pi_capt / pi_wild))
print("pooling adds {:,.0f} segregating sites at full sample size".format(
    spectra["pooled full (52,12)"].S() - spectra["wild-caught (32,12)"].S()))
