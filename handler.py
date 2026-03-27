import runpod
import os
import sys
import json
import tempfile
import shutil
import base64
import traceback
from pathlib import Path


SUPPORTED_SUFFIXES = {".npy", ".csv", ".json", ".dat", ".xml", ".dcm"}


def load_signal(path: Path, suffix: str):
    """Load ECG data and save as .npy file. Returns path to .npy file."""
    import numpy as np

    if suffix == ".npy":
        arr = np.load(str(path))
    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path, header=None)
        arr = df.values.astype(np.float32)
    elif suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "signal" in data:
            arr = np.array(data["signal"], dtype=np.float32)
        else:
            raise ValueError("JSON must be list or dict with 'signal' key")
    elif suffix == ".dat":
        import wfdb
        record = wfdb.rdrecord(str(path.with_suffix("")))
        arr = record.p_signal.T.astype(np.float32)
    elif suffix == ".xml":
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        leads = []
        for wave in root.iter("WaveformData"):
            vals = [float(v) for v in wave.text.strip().split(",")]
            leads.append(vals)
        arr = np.array(leads, dtype=np.float32)
    elif suffix == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.waveform_array(0).T.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Supported: {', '.join(SUPPORTED_SUFFIXES)}")

    # Ensure shape is (leads, samples) — at least 1D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    out_path = path.parent / "ecg.npy"
    np.save(str(out_path), arr)
    return out_path


def run_pipeline(npy_path: Path, work_dir: Path) -> dict:
    import numpy as np
    import pandas as pd

    sys.path.insert(0, "/workspace/DeepECG_Docker")
    sys.path.insert(0, "/workspace/DeepECG_Docker/fairseq-signals")
    os.environ["MPLBACKEND"] = "Agg"

    from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, MODEL_MAPPING
    from utils.analysis_pipeline import AnalysisPipeline
    from main import (
        create_preprocessing_dataframe,
        create_analysis_dataframe,
        validate_dataframe,
    )

    data_dir = work_dir / "data"
    out_dir  = work_dir / "output"
    prep_dir = work_dir / "preprocessed"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    # Copy ECG file into data dir
    dest = data_dir / "ecg.npy"
    shutil.copy(str(npy_path), str(dest))

    hf_token = os.environ.get("HF_TOKEN", "")
    errors   = []

    # Build manifest DataFrame
    manifest_df = pd.DataFrame([{
        "patient_id":                  "pt001",
        "record_name":                 "ecg",
        "ecg_machine_diagnosis":       0,
        "77_classes_ecg_file_name":    "ecg.npy",
        "afib_5y":                     0,
        "afib_ecg_file_name":          "ecg.npy",
        "lvef_40":                     0,
        "lvef_40_ecg_file_name":       "ecg.npy",
        "lvef_50":                     0,
        "lvef_50_ecg_file_name":       "ecg.npy",
    }])

    existing_diagnosis_columns, existing_file_columns = validate_dataframe(
        df=manifest_df,
        diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS,
    )

    df_preprocessing, df_missing = create_preprocessing_dataframe(
        df=manifest_df,
        existing_file_columns=existing_file_columns,
        ecg_path=str(data_dir),
    )

    _ = AnalysisPipeline.save_and_preprocess_data(
        df=df_preprocessing,
        output_folder=str(out_dir),
        preprocessing_folder=str(prep_dir),
        preprocessing_n_workers=1,
        errors=errors,
    )

    results = {}
    for diagnosis_column in existing_diagnosis_columns:
        try:
            ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]
            df_analysis = create_analysis_dataframe(
                df=manifest_df,
                diagnosis_column=diagnosis_column,
                ecg_file_column=ecg_file_column,
                preprocessing_folder=str(prep_dir),
            )
            metrics, df_probabilities = AnalysisPipeline.run_analysis(
                df=df_analysis,
                batch_size=1,
                diagnosis_classifier_device="cuda",
                signal_processing_device="cpu",          # ← CPU avoids CUDA kernel mismatch
                signal_processing_model_name=MODEL_MAPPING[diagnosis_column].get("efficientnet"),
                diagnosis_classifier_model_name=MODEL_MAPPING[diagnosis_column].get("bert"),
                hugging_face_api_key=hf_token,
                errors=errors,
            )
            if df_probabilities is not None and not df_probabilities.empty:
                prob_cols = [c for c in df_probabilities.columns if "prob" in c.lower() or "pred" in c.lower()]
                results[diagnosis_column] = df_probabilities[prob_cols].iloc[0].to_dict() if prob_cols else {}
        except Exception as e:
            errors.append(f"{diagnosis_column}: {str(e)}")
            traceback.print_exc()

    return {"results": results, "errors": errors}


def handler(job):
    job_input = job["input"]

    ecg_b64  = job_input.get("ecg_base64", "")
    filename = job_input.get("filename", "ecg.npy")

    if not ecg_b64:
        return {"error": "ecg_base64 is required"}

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return {
            "error": (
                f"Unsupported file type '{suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_SUFFIXES))}. "
                "Note: image scans (jpg/png/pdf) are not supported — raw ECG signal files required."
            )
        }

    work_dir = Path(tempfile.mkdtemp())
    try:
        raw_path = work_dir / filename
        raw_path.write_bytes(base64.b64decode(ecg_b64))

        npy_path = load_signal(raw_path, suffix)
        return run_pipeline(npy_path, work_dir)

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


runpod.serverless.start({"handler": handler})
