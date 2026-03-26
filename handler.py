import runpod, os, sys, json, tempfile, shutil, base64, traceback
from pathlib import Path

sys.path.insert(0, "/workspace/DeepECG_Docker")
sys.path.insert(0, "/workspace/DeepECG_Docker/fairseq-signals")
os.environ["MPLBACKEND"] = "Agg"


def load_signal(path, suffix):
    import numpy as np
    try:
        if suffix == ".npy":    return np.load(str(path))
        elif suffix == ".csv":
            import pandas as pd; return pd.read_csv(path).values.astype("float32")
        elif suffix == ".json":
            data = json.loads(path.read_text())
            return np.array(data["signal"] if "signal" in data else data, dtype="float32")
        elif suffix in (".jpg",".jpeg",".png",".pdf"):
            from PIL import Image; import numpy as np
            return np.array(Image.open(path).convert("L"), dtype="float32")
        elif suffix == ".dat":
            import wfdb; rec = wfdb.rdrecord(str(path.with_suffix("")))
            return rec.p_signal.T.astype("float32")
        elif suffix == ".xml":
            import xml.etree.ElementTree as ET
            samples = []
            for e in ET.parse(path).iter():
                if e.text:
                    try: samples.append(float(e.text))
                    except: pass
            return np.array(samples, dtype="float32").reshape(1, -1)
        elif suffix == ".dcm":
            import pydicom; return pydicom.dcmread(str(path)).pixel_array.astype("float32")
    except Exception as e:
        print(f"load_signal error ({suffix}): {e}")
    return None


def run_pipeline(npy_path, work_dir):
    import numpy as np, pandas as pd

    from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, MODEL_MAPPING
    from utils.analysis_pipeline import AnalysisPipeline
    from main import create_preprocessing_dataframe, create_analysis_dataframe

    ecg_dir           = work_dir / "ecg";           ecg_dir.mkdir(exist_ok=True)
    output_dir        = work_dir / "output";        output_dir.mkdir(exist_ok=True)
    preprocessing_dir = work_dir / "preprocessing"; preprocessing_dir.mkdir(exist_ok=True)

    ecg_filename = "ecg.npy"
    shutil.copy(str(npy_path), str(ecg_dir / ecg_filename))

    sig = np.load(str(npy_path))
    print(f"[DeepECG] signal shape: {sig.shape}, dtype: {sig.dtype}")

    # Build DataFrame — all diagnosis file columns point to the same ECG file
    file_cols = list(DIAGNOSIS_TO_FILE_COLUMNS.values())
    row = {"patient_id": "pt001"}
    for col in file_cols:
        row[col] = ecg_filename
    df = pd.DataFrame([row])

    errors = []

    # Step 1 — Preprocessing
    print("[DeepECG] Preprocessing...")
    df_preprocessing, df_missing = create_preprocessing_dataframe(
        df=df,
        existing_file_columns=file_cols,
        ecg_path=str(ecg_dir),
    )
    print(f"[DeepECG] df_preprocessing shape: {df_preprocessing.shape}")

    preprocessed_df = AnalysisPipeline.save_and_preprocess_data(
        df=df_preprocessing,
        output_folder=str(output_dir),
        preprocessing_folder=str(preprocessing_dir),
        preprocessing_n_workers=1,
        errors=errors,
    )

    if preprocessed_df is None:
        return {"status": "error", "model": "deepecg", "error": "Preprocessing failed", "errors": errors}

    # Step 2 — Analysis per diagnosis
    all_findings = {}
    hf_key = os.environ.get("HF_TOKEN", "")

    for diagnosis_column, ecg_file_column in DIAGNOSIS_TO_FILE_COLUMNS.items():
        try:
            df_analysis = create_analysis_dataframe(
                df=preprocessed_df,
                diagnosis_column=diagnosis_column,
                ecg_file_column=ecg_file_column,
                preprocessing_folder=str(preprocessing_dir),
            )
            if df_analysis is None or len(df_analysis) == 0:
                print(f"[DeepECG] No analysis rows for {diagnosis_column}")
                continue

            models     = MODEL_MAPPING.get(diagnosis_column, {})
            sig_model  = models.get("wcr", "")
            diag_model = models.get("efficientnet", "")
            if not sig_model or not diag_model:
                continue

            results, result_df = AnalysisPipeline.run_analysis(
                df=df_analysis,
                batch_size=1,
                diagnosis_classifier_device=0,
                signal_processing_device=0,
                signal_processing_model_name=sig_model,
                diagnosis_classifier_model_name=diag_model,
                hugging_face_api_key=hf_key,
                errors=errors,
            )

            if results:
                all_findings[diagnosis_column] = results
            if result_df is not None:
                all_findings[f"{diagnosis_column}_df"] = result_df.to_dict(orient="records")

        except Exception as e:
            traceback.print_exc()
            errors.append(f"{diagnosis_column}: {str(e)}")

    return {"status": "ok", "model": "deepecg", "findings": all_findings, "errors": errors}


def handler(job):
    job_input = job["input"]
    try:
        ecg_b64  = job_input.get("ecg_base64", "")
        filename = job_input.get("filename", "ecg.npy")
        suffix   = Path(filename).suffix.lower()

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
