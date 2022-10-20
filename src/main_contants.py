"""
FILENAMES
"""

RAW_CSV_FILENAME: str = "../data/Dataset-Unicauca-Version2-87Atts.csv"
FILENAME_CLEANED_DATA: str = "../data/dataset_cleaned.csv"

"""
LISTS
"""
UNUSED_COLS: list[str] = [
    "Flow.ID",
    "Timestamp",
    "ProtocolName",
    "Source.IP",
    "Destination.IP",
]

"""
ML CONSTANTS
"""

TARGET: str = "L7Protocol"

SEED: int = 1  # reproducibility

RANDOM_FOREST_ESTIMATORS: int = 50
