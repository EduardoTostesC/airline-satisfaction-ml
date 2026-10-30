"""Validação do conteúdo gravado e alinhamento explícito por ID."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_csv, validate_ids


def validate_probabilities(values, expected_length):
    values = np.asarray(values, dtype=float)
    if values.shape != (expected_length,):
        raise ValueError("Quantidade ou dimensão de probabilidades incompatível.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilidades devem ser finitas entre 0 e 1.")
    return values


def validate_submission(frame, sample, test_ids, ident="id", target="satisfaction"):
    if list(frame.columns) != [ident, target]:
        raise ValueError("Cabeçalho da submissão incorreto ou índice extra.")
    validate_ids(frame, ident, "submission")
    if len(frame) != len(sample) or set(frame[ident]) != set(test_ids):
        raise ValueError("IDs da submissão não correspondem ao teste.")
    if frame[ident].tolist() != sample[ident].tolist():
        raise ValueError("Ordem da submissão diferente do sample.")
    validate_probabilities(pd.to_numeric(frame[target], errors="raise"), len(sample))


def write_submission(test_ids, probabilities, sample, path, ident="id", target="satisfaction"):
    values = validate_probabilities(probabilities, len(test_ids))
    if test_ids.duplicated().any() or set(test_ids) != set(sample[ident]):
        raise ValueError("IDs não permitem alinhamento unívoco entre teste e sample.")
    predictions = pd.Series(values, index=test_ids.to_numpy())
    result = pd.DataFrame({ident: sample[ident], target: predictions.loc[sample[ident]].to_numpy()})
    validate_submission(result, sample, test_ids, ident, target)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.17g")
    validate_submission(read_csv(path, ident), sample, test_ids, ident, target)
    return path
