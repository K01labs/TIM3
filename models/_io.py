"""Shared prediction-file I/O for models.

Every model writes one Parquet prediction file per sample with the same schema,
so the writer lives here once.

Prediction schema:
    t      : float32        — query timestamp (copied from the query set)
    s      : int32          — signal ID
    j      : int32          — channel index
    x_pred : list[float32]  — predicted value vector of shape (d_s,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PRED_SCHEMA = pa.schema(
    [
        pa.field("t", pa.float32()),
        pa.field("s", pa.int32()),
        pa.field("j", pa.int32()),
        pa.field("x_pred", pa.list_(pa.float32())),
    ]
)


def write_predictions(pred_df: pd.DataFrame, out_path: Path) -> None:
    """Write a predictions DataFrame to Parquet with the benchmark schema.

    Args:
        pred_df: DataFrame with columns ``(t, s, j, x_pred)`` where ``x_pred``
            is a numpy float32 array. May be empty (an empty file with the
            correct schema is written, e.g. for cross-modal LOCF).
        out_path: Destination path; parent directories are created if absent.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(pred_df) == 0:
        table = pa.table(
            {
                "t": pa.array([], type=pa.float32()),
                "s": pa.array([], type=pa.int32()),
                "j": pa.array([], type=pa.int32()),
                "x_pred": pa.array([], type=pa.list_(pa.float32())),
            },
            schema=PRED_SCHEMA,
        )
    else:
        table = pa.table(
            {
                "t": pa.array(
                    pred_df["t"].to_numpy(dtype=np.float32), type=pa.float32()
                ),
                "s": pa.array(pred_df["s"].to_numpy(dtype=np.int32), type=pa.int32()),
                "j": pa.array(pred_df["j"].to_numpy(dtype=np.int32), type=pa.int32()),
                "x_pred": pa.array(
                    [np.asarray(arr).tolist() for arr in pred_df["x_pred"]],
                    type=pa.list_(pa.float32()),
                ),
            },
            schema=PRED_SCHEMA,
        )

    pq.write_table(table, out_path)
