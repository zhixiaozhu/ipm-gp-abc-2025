"""
Download the precomputed results needed to reproduce the full analyses, and
extract them into the two case-study folders.

These files are large, so are not stored in the repository. They are attached
to a release on the repository's GitHub Releases page. NO login is needed.

Running this is ONLY necessary to reproduce the published results; to try the
method or read through the code, the toy example (toy_example/vignette.ipynb)
needs NONE of them.

    python download_full_results.py          # everything needed for the results
    python download_full_results.py --all    # also the ABC efficiency diagnostics

Notice that, the cached GP posteriors under true_popu/mcmc/Lm/ are NOT part of this download.
They exceed 50 GB in total, and are regenerated from the MCMC samples by
s1_cache_produce.py (simulation study) or r2_cache_producing_and_testing.py (real
study), which MUST be run before the ABC sampling and prediction scripts.
"""


import argparse
import os
import sys
import tarfile
import urllib.error
import urllib.request

REPO = "zhixiaozhu/ipm-gp-abc-2025"
TAG = "data-v1"
SIM = "simulation_case_study"
REAL = "real_case_study/ABC model fitting"
SS = "real_case_study/ss_selection"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = [
    ("simulation_results.tar.gz", SIM,
     "true_popu/glm_mle_true_population/glm_mle_true_population.pkl", False),
    ("real_true_popu.tar.gz", REAL,
     "true_popu/mcmc", False),
    ("real_abc_details.tar.gz", REAL,
     "ABC_details/p_index_10.pkl", False),
    ("real_ss_mcmc_samples.tar.gz", SS,
     "mcmc_samples", False),
    ("real_abc_particles_summary.tar.gz", REAL,
     "ABC_details/s0_random", True),
]


def _progress(count, block_size, total):
    if total > 0:
        pct = min(100.0, 100.0 * count * block_size / total)
        sys.stdout.write(f"\r {pct:5.1f}%  of {total/1e6:.0f} MB")
    else:
        sys.stdout.write(f"\r {count * block_size / 1e6:.0f} MB")
    sys.stdout.flush()


def _safe_extract(tar, path):
    # refuse archive members that would be written outside the target folder
    base = os.path.abspath(path)
    for member in tar.getmembers():
        dest = os.path.abspath(os.path.join(path, member.name))
        if dest != base and not dest.startswith(base + os.sep):
            raise RuntimeError(f"unsafe path in archive: {member.name}")
    tar.extractall(path)


def fetch(asset, target, marker):
    dest_dir = os.path.join(HERE, target)
    if os.path.exists(os.path.join(dest_dir, marker)):
        print(f"{asset}\n    already present, skipping")
        return

    url = f"https://github.com/{REPO}/releases/download/{TAG}/{asset}"
    tmp = os.path.join(HERE, asset)
    print(f"{asset}")
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
    except urllib.error.HTTPError as e:
        sys.exit(f"\n Could not download (HTTP {e.code}).\n"
                 f"Tried {url}\n"
                 f"Check that the release '{TAG}' exists, that it has an asset "
                 f"named '{asset}', and that the repository is public.")

    print(f"\n Extracting into {target}/")
    with tarfile.open(tmp, "r:gz") as tar:
        _safe_extract(tar, dest_dir)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also download the ABC particle summaries, which are only "
                         "needed for the efficiency figures")
    args = ap.parse_args()

    wanted = [a for a in ASSETS if args.all or not a[3]]
    skipped = len(ASSETS) - len(wanted)

    for asset, target, marker, _ in wanted:
        fetch(asset, target, marker)

    print("\nDone.")
    if skipped:
        print("The ABC particle summaries were not downloaded. They are only needed to "
              "reproduce the ABC efficiency figures (r7_efficiency.ipynb and "
              "s5_efficiency.ipynb); run with --all to include them.")


if __name__ == "__main__":
    main()
