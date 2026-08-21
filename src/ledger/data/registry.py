"""Declarative dataset registry: the single source of truth for what
datasets LEDGER knows about, where they come from, what files they ship,
and what license governs each.

Nothing downloads on import. This module is pure metadata; acquisition and
checksum verification live in :mod:`ledger.data.fetch`.

Checksums are intentionally left unpinned (``expected_sha256=None``) for
datasets gated behind an account (Kaggle, a GitHub release) where the
publisher does not itself publish a stable, citable digest: fabricating one
would be worse than not having it. The first verified fetch on a given
machine reports the actual digest so an operator can pin it in a local
override (see :func:`ledger.data.fetch.verify_dataset_files`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetFile:
    """One file expected inside a dataset's raw directory."""

    filename: str
    description: str
    expected_sha256: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to acquire, verify, and describe one dataset."""

    name: str
    role: str
    license_name: str
    source_description: str
    required_credentials: tuple[str, ...]
    fetch_instructions: str
    files: tuple[DatasetFile, ...] = field(default_factory=tuple)


_KAGGLE_CREDENTIAL_STEPS = (
    "1. Create a Kaggle account if you do not have one, then go to "
    "https://www.kaggle.com/settings -> 'API' -> 'Create New Token'. This "
    "downloads kaggle.json.\n"
    "2. Place it at ~/.kaggle/kaggle.json (chmod 600), or export "
    "KAGGLE_USERNAME and KAGGLE_KEY as environment variables.\n"
    "3. pip install kaggle\n"
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "elliptic": DatasetSpec(
        name="elliptic",
        role=(
            "Primary graph benchmark. 203,769 nodes, 234,355 edges, 166 "
            "features, 49 time steps, licit/illicit/unknown labels."
        ),
        license_name="CC BY-NC-SA 4.0 (per Kaggle listing; non-commercial use only)",
        source_description="Kaggle: ellipticco/elliptic-data-set",
        required_credentials=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        fetch_instructions=(
            _KAGGLE_CREDENTIAL_STEPS
            + "4. kaggle datasets download -d ellipticco/elliptic-data-set "
            "-p data/raw/elliptic --unzip\n"
            "5. Confirm the files below are present, then re-run "
            "`ledger data fetch elliptic` to verify checksums.\n"
            "Note: verify the current listing slug on kaggle.com before "
            "downloading; dataset listings occasionally move."
        ),
        files=(
            DatasetFile(
                "elliptic_txs_features.csv",
                "166 columns per transaction node: txId, time step, 165 "
                "anonymised features. No header row.",
            ),
            DatasetFile(
                "elliptic_txs_classes.csv",
                "txId -> class in {'1' (illicit), '2' (licit), 'unknown'}.",
            ),
            DatasetFile(
                "elliptic_txs_edgelist.csv",
                "txId1,txId2 directed edges between transaction nodes.",
            ),
        ),
    ),
    "elliptic_pp": DatasetSpec(
        name="elliptic_pp",
        role="Actor-level accounts, enables ring-level evaluation.",
        license_name="CC BY-NC-SA 4.0 (see the Elliptic++ repository for current terms)",
        source_description="Elliptic++ release (git-disl/EllipticPlusPlus on GitHub)",
        required_credentials=(),
        fetch_instructions=(
            "1. Go to the Elliptic++ dataset repository "
            "(git-disl/EllipticPlusPlus on GitHub) and download the "
            "actor-level release assets referenced in its README.\n"
            "2. Extract the archive into data/raw/elliptic_pp/.\n"
            "3. Confirm the files below are present, then re-run "
            "`ledger data fetch elliptic_pp` to verify checksums.\n"
            "Note: this dataset is not distributed via Kaggle; no API "
            "credentials are required, but the release asset URL changes "
            "between versions — check the repository README for the "
            "current link."
        ),
        files=(
            DatasetFile(
                "wallets_features_classes_combined.csv",
                "Actor (wallet address) level features plus class label "
                "in {1 (illicit), 2 (licit), 3 (unknown)}.",
            ),
            DatasetFile(
                "AddrAddr_edgelist.csv",
                "input_address,output_address actor-to-actor transaction "
                "edges, aggregated from the underlying transaction graph.",
            ),
        ),
    ),
    "ibm_aml": DatasetSpec(
        name="ibm_aml",
        role=(
            "Synthetic inter-bank transactions with explicit "
            "laundering-pattern labels; drives the federated experiment."
        ),
        license_name="See Kaggle listing (IBM synthetic data, released for research use)",
        source_description=("Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml"),
        required_credentials=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        fetch_instructions=(
            _KAGGLE_CREDENTIAL_STEPS + "4. kaggle datasets download -d "
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml "
            "-p data/raw/ibm_aml --unzip\n"
            "5. The release ships several size variants (HI-Small, "
            "HI-Medium, HI-Large, LI-Small, LI-Medium, LI-Large); LEDGER "
            "defaults to HI-Small_Trans.csv for tractability. Confirm the "
            "file below is present, then re-run "
            "`ledger data fetch ibm_aml` to verify checksums."
        ),
        files=(
            DatasetFile(
                "HI-Small_Trans.csv",
                "Timestamp, From Bank, Account, To Bank, Account.1, Amount "
                "Received, Receiving Currency, Amount Paid, Payment "
                "Currency, Payment Format, Is Laundering.",
            ),
        ),
    ),
    "paysim": DatasetSpec(
        name="paysim",
        role="6.3M mobile-money transactions; scale testing.",
        license_name="CC0: Public Domain (per Kaggle listing)",
        source_description="Kaggle: ealaxi/paysim1",
        required_credentials=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        fetch_instructions=(
            _KAGGLE_CREDENTIAL_STEPS + "4. kaggle datasets download -d ealaxi/paysim1 "
            "-p data/raw/paysim --unzip\n"
            "5. Confirm the file below is present, then re-run "
            "`ledger data fetch paysim` to verify checksums."
        ),
        files=(
            DatasetFile(
                "PS_20174392719_1491204439457_log.csv",
                "step, type, amount, nameOrig, oldbalanceOrg, "
                "newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, "
                "isFraud, isFlaggedFraud.",
            ),
        ),
    ),
    "baf": DatasetSpec(
        name="baf",
        role="Contains protected attributes; drives the bias audit.",
        license_name="CC BY-NC-SA 4.0 (per NeurIPS 2022 Datasets and Benchmarks release)",
        source_description=(
            "Kaggle: sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            "(NeurIPS 2022 Bank Account Fraud)"
        ),
        required_credentials=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        fetch_instructions=(
            _KAGGLE_CREDENTIAL_STEPS + "4. kaggle datasets download -d "
            "sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            "-p data/raw/baf --unzip\n"
            "5. The release ships Base.csv plus six distribution-shifted "
            "variants (Variant I-V); LEDGER's adapter uses Base.csv. "
            "Confirm it is present, then re-run `ledger data fetch baf` "
            "to verify checksums."
        ),
        files=(
            DatasetFile(
                "Base.csv",
                "One row per account application: fraud_bool label plus "
                "~30 numeric/categorical features including customer_age "
                "(the protected attribute used for the Phase 7 bias "
                "audit, per the dataset's own fairness benchmark).",
            ),
        ),
    ),
    "ieee_cis": DatasetSpec(
        name="ieee_cis",
        role="Secondary tabular baseline.",
        license_name="See Kaggle competition rules (ieee-fraud-detection)",
        source_description="Kaggle competition: ieee-fraud-detection",
        required_credentials=("KAGGLE_USERNAME", "KAGGLE_KEY"),
        fetch_instructions=(
            _KAGGLE_CREDENTIAL_STEPS + "4. Accept the competition rules at "
            "https://www.kaggle.com/c/ieee-fraud-detection/rules, then:\n"
            "   kaggle competitions download -c ieee-fraud-detection "
            "-p data/raw/ieee_cis\n"
            "5. Unzip train_transaction.csv.zip and "
            "train_identity.csv.zip into data/raw/ieee_cis/, then re-run "
            "`ledger data fetch ieee_cis` to verify checksums."
        ),
        files=(
            DatasetFile(
                "train_transaction.csv",
                "TransactionID, isFraud, TransactionDT, TransactionAmt, "
                "ProductCD, card1-6, addr1-2, and C/D/M/V feature columns.",
            ),
            DatasetFile(
                "train_identity.csv",
                "TransactionID, id_01-38, DeviceType, DeviceInfo — joined "
                "to train_transaction.csv on TransactionID.",
            ),
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset's spec by name, raising a clear error if unknown."""
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        allowed = sorted(DATASET_REGISTRY)
        raise KeyError(f"Unknown dataset {name!r}. Registered datasets: {allowed}") from exc


__all__ = ["DatasetFile", "DatasetSpec", "DATASET_REGISTRY", "get_dataset_spec"]
