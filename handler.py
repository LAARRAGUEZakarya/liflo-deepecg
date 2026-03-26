import runpod, os, sys, json, tempfile, shutil, base64, traceback
from pathlib import Path


def load_signal(path, suffix):
    import numpy as np
    try:
        if suffix == ".npy":
            return np.load(str(path))
        elif suffix == ".csv":
            import pandas as pd
            return pd.read_csv(path).values.astype("float32")
        elif suffix == ".json":
            data = json.loads(path.read_text())
            return np.array(data["signal"] if "signal" in data else data, dtype="float32")
        elif suffix in (".jpg", ".jpeg", ".png", ".pdf"):
            from PIL import Image
            return np.array(Image.open(path).convert("L"), dtype="float32")
        elif suffix == ".dat":
            import wfdb
            rec = wfdb.rdrecord(str(path.with_suffix("")))
            return rec.p_signal.T.astype("float32")
        elif suffix == ".xml":
            import xml.etree.ElementTree as ET
            import numpy as np
            samples = []
            for e in ET.parse(path).iter():
                if e.text:
                    try:
                        samples.append(float(e.text))
                    except:
                        pass
            return np.array(samples, dtype="float32").reshape(1, -1)
        elif suffix == ".dcm":
            import pydicom
            return pydicom.dcmread(str(path)).pixel_array.astype("float32")
    except Exception as e:
        print(f"load_signal error ({suffix}): {e}")
    return None


def run_pipeline(npy_path, work_dir):
    import numpy as np
    import pandas as pd
    import torch

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
    out_dir  = work_dir / "out"
    prep_dir = work_dir / "prep"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    prep_dir.mkdir(exist_ok=True)

    shutil.copy(str(npy_path), str(data_dir / "ecg.npy"))

    sig = np.load(str(npy_path))
    print(f"[DeepECG] signal shape: {sig.shape}, dtype: {sig.dtype}")

    # Build manifest DataFrame — use 0 as placeholder label for inference
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

    # Step 1 — validate columns
    existing_diagnosis_columns, existing_file_columns = validate_dataframe(
        df=manifest_df,
        diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS,
    )
    print(f"[DeepECG] diagnosis cols: {existing_diagnosis_columns}")

    # Step 2 — create preprocessing DataFrame (pass DataFrame, NOT string)
    df_preprocessing, df_missing = create_preprocessing_dataframe(
        df=manifest_df,
        existing_file_columns=existing_file_columns,
        ecg_path=str(data_dir),
    )
    print(f"[DeepECG] preprocessing rows: {len(df_preprocessing)}, missing: {len(df_missing)}")

    if df_preprocessing.empty:
        return {"error": "No valid ECG files found", "missing": df_missing.to_dict()}

    # Step 3 — preprocess ECG → .base64 files
    preprocessed_df = AnalysisPipeline.save_and_preprocess_data(
        df=df_preprocessing,
        output_folder=str(out_dir),
        preprocessing_folder=str(prep_dir),
        preprocessing_n_workers=1,
        errors=[],
    )

    if preprocessed_df is None:
        return {"error": "ECG preprocessing failed (bad signal format?)"}

    # Step 4 — run analysis per diagnosis
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get("HF_TOKEN", "")
    findings = {}
    errors   = []

    for diagnosis_column in existing_diagnosis_columns:
        ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]

        df_analysis = create_analysis_dataframe(
            df=manifest_df,
            diagnosis_column=diagnosis_column,
            ecg_file_column=ecg_file_column,
            preprocessing_folder=str(prep_dir),
        )

        if df_analysis.empty:
            print(f"[DeepECG] No analysis data for {diagnosis_column}")
            continue

        signal_model = MODEL_MAPPING[diagnosis_column].get("efficientnet")
        bert_model   = MODEL_MAPPING[diagnosis_column].get("bert")

        if not signal_model or not bert_model:
            continue

        metrics, df_probabilities = AnalysisPipeline.run_analysis(
            df=df_analysis,
            batch_size=1,
            diagnosis_classifier_device=device,
            signal_processing_device=device,
            signal_processing_model_name=signal_model,
            diagnosis_classifier_model_name=bert_model,
            hugging_face_api_key=hf_token,
            errors=errors,
        )

        if df_probabilities is not None:
            findings[diagnosis_column] = df_probabilities.to_dict(orient="records")

    return {
        "status":   "ok",
        "model":    "deepecg",
        "findings": findings,
        "errors":   errors,
    }


def handler(job):
    job_input = job["input"]
    try:
        ecg_b64  = job_input.get("ecg_base64", "")
        filename = job_input.get("filename", "ecg.npy")
        suffix   = Path(filename).suffix.lower()

        if not ecg_b64:
            return {"error": "No ECG file provided"}

        raw      = base64.b64decode(ecg_b64)
        work_dir = Path(tempfile.mkdtemp())
        ecg_path = work_dir / f"ecg{suffix}"
        ecg_path.write_bytes(raw)

        import numpy as np
        signal = load_signal(ecg_path, suffix)
        if signal is None:
            return {"error": f"Cannot parse {suffix}"}

        npy_path = work_dir / "ecg.npy"
        np.save(str(npy_path), signal)

        result = run_pipeline(npy_path, work_dir)
        shutil.rmtree(work_dir, ignore_errors=True)
        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
