import runpod
import os
import sys
import json
import tempfile
import shutil
import base64
import traceback
from pathlib import Path

IMAGE_SUFFIXES  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
SIGNAL_SUFFIXES = {".npy", ".csv", ".json", ".dat", ".xml", ".dcm"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | SIGNAL_SUFFIXES
SKIP_DIAGNOSES = {"ecg_machine_diagnosis"}


def image_to_signal(image_path: Path):
    """Digitize an ECG image into a (1, N) float32 signal array."""
    import numpy as np
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image as PILImage
        pil = PILImage.open(image_path).convert("L")
        img = np.array(pil, dtype=np.uint8)

    # Resize to fixed width for consistent sampling
    target_w = 5000
    h, w = img.shape
    if w != target_w:
        img = cv2.resize(img, (target_w, h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape

    # Invert so ECG trace is bright (trace is dark on white paper)
    inv = 255 - img.astype(np.float32)

    # Threshold to isolate trace
    threshold = inv.max() * 0.35
    binary = (inv > threshold).astype(np.float32)

    signal = np.zeros(w, dtype=np.float32)
    for x in range(w):
        col = binary[:, x]
        rows = np.where(col > 0)[0]
        if len(rows) > 0:
            signal[x] = h - float(np.mean(rows))
        else:
            signal[x] = signal[x - 1] if x > 0 else 0.0

    # Normalize to unit variance, zero mean
    mean, std = signal.mean(), signal.std()
    if std > 1e-6:
        signal = (signal - mean) / std

    return signal.reshape(1, -1).astype(np.float32)


def load_signal(path: Path, suffix: str) -> Path:
    """Load / convert any supported format to a .npy file. Returns path to .npy."""
    import numpy as np

    if suffix in IMAGE_SUFFIXES:
        arr = image_to_signal(path)

    elif suffix == ".npy":
        arr = np.load(str(path))

    elif suffix == ".csv":
        import pandas as pd
        arr = pd.read_csv(path, header=None).values.astype(np.float32)

    elif suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "signal" in data:
            arr = np.array(data["signal"], dtype=np.float32)
        else:
            raise ValueError("JSON must be a list or dict with a 'signal' key")

    elif suffix == ".dat":
        import wfdb
        record = wfdb.rdrecord(str(path.with_suffix("")))
        arr = record.p_signal.T.astype(np.float32)

    elif suffix == ".xml":
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        leads = []
        for wave in tree.getroot().iter("WaveformData"):
            leads.append([float(v) for v in wave.text.strip().split(",")])
        arr = np.array(leads, dtype=np.float32)

    elif suffix == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.waveform_array(0).T.astype(np.float32)

    else:
        raise ValueError(f"Unsupported format: {suffix}")

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
    for d in (data_dir, out_dir, prep_dir):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(str(npy_path), str(data_dir / "ecg.npy"))

    hf_token = os.environ.get("HF_TOKEN", "")
    errors   = []

    manifest_df = pd.DataFrame([{
        "patient_id":               "pt001",
        "record_name":              "ecg",
        "ecg_machine_diagnosis":    0,
        "77_classes_ecg_file_name": "ecg.npy",
        "afib_5y":                  0,
        "afib_ecg_file_name":       "ecg.npy",
        "lvef_40":                  0,
        "lvef_40_ecg_file_name":    "ecg.npy",
        "lvef_50":                  0,
        "lvef_50_ecg_file_name":    "ecg.npy",
    }])

    existing_diagnosis_columns, existing_file_columns = validate_dataframe(
        df=manifest_df,
        diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS,
    )

    df_preprocessing, _ = create_preprocessing_dataframe(
        df=manifest_df,
        existing_file_columns=existing_file_columns,
        ecg_path=str(data_dir),
    )

    AnalysisPipeline.save_and_preprocess_data(
        df=df_preprocessing,
        output_folder=str(out_dir),
        preprocessing_folder=str(prep_dir),
        preprocessing_n_workers=1,
        errors=errors,
    )

    results = {}
    for diagnosis_column in existing_diagnosis_columns:
        if diagnosis_column in SKIP_DIAGNOSES:
            continue
        try:
            ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]
            df_analysis = create_analysis_dataframe(
                df=manifest_df,
                diagnosis_column=diagnosis_column,
                ecg_file_column=ecg_file_column,
                preprocessing_folder=str(prep_dir),
            )
            _, df_prob = AnalysisPipeline.run_analysis(
                df=df_analysis,
                batch_size=1,
                diagnosis_classifier_device="cuda",
                signal_processing_device="cpu",
                signal_processing_model_name=MODEL_MAPPING[diagnosis_column].get("efficientnet"),
                diagnosis_classifier_model_name=MODEL_MAPPING[diagnosis_column].get("bert"),
                hugging_face_api_key=hf_token,
                errors=errors,
            )
            if df_prob is not None and not df_prob.empty:
                prob_cols = [c for c in df_prob.columns if "prob" in c.lower() or "pred" in c.lower()]
                results[diagnosis_column] = df_prob[prob_cols].iloc[0].to_dict() if prob_cols else {}
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
        return {"error": f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"}

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
