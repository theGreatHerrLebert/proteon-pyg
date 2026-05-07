"""proteon-pyg — proteon structural features as PyTorch Geometric Data tensors."""

from proteon_pyg.features import (
    DSSP_CLASSES,
    ProteonFeatures,
    decode_dssp,
    encode_dssp,
    proteon_pyg_data,
)

__all__ = [
    "DSSP_CLASSES",
    "ProteonFeatures",
    "decode_dssp",
    "encode_dssp",
    "proteon_pyg_data",
]

__version__ = "0.0.1"
