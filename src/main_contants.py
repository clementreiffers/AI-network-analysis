"""
FILENAMES
"""
DATA_PATH: str = "../data"
RAW_CSV_FILENAME: str = f"{DATA_PATH}/Dataset-Unicauca-Version2-87Atts.csv"
FILENAME_CLEANED_DATA: str = f"{DATA_PATH}/dataset_cleaned.csv"
STATISTICS_CSV: str = f"{DATA_PATH}/statistics.csv"
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
