import runpod, os, sys, json, tempfile, shutil, base64, traceback
from pathlib import Path

sys.path.insert(0, "/workspace/DeepECG_Docker")
sys.path.insert(0, "/workspace/DeepECG_Docker/fairseq-signals")
os.environ["MPLBACKEND"] = "Agg"


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
            samples = []
            for e in ET.parse(path).iter():
                if e.text:
                    try: samples.append(float(e.text))
                    except: pass
            return np.array(samples, dtype="float32").reshape(1, -1)
        elif suffix == ".dcm":
            import pydicom
            return pydicom.dcmread(str(path)).pixel_array.astype("float32")
    except Exception as e:
        print(f"load_signal error ({suffix}): {e}")
    return None


def run_pipeline(npy_path, work_dir, hf_token):
    import numpy as np
    import pandas as pd
    from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, MODEL_MAPPING
    from utils.analysis_pipeline import AnalysisPipeline
    from main import create_preprocessing_dataframe, create_analysis_dataframe, validate_dataframe

    data_dir = work_dir / "data"
    out_dir  = work_dir / "output"
    pre_dir  = work_dir / "preprocessing"
    for d in [data_dir, out_dir, pre_dir]:
        d.mkdir(exist_ok=True)

    ecg_filename = "ecg.npy"
    shutil.copy(str(npy_path), str(data_dir / ecg_filename))

    # Build dataframe with all diagnosis + file columns
    df = pd.DataFrame([{
        "patient_id":               "pt001",
        "ecg_machine_diagnosis":    "unknown",
        "77_classes_ecg_file_name": ecg_filename,
        "afib_5y":                  0,
        "afib_ecg_file_name":       ecg_filename,
        "lvef_40":                  0,
        "lvef_40_ecg_file_name":    ecg_filename,
        "lvef_50":                  0,
        "lvef_50_ecg_file_name":    ecg_filename,
    }])

    existing_diagnosis_columns, existing_file_columns = validate_dataframe(
        df=df, diagnosis_to_file_columns=DIAGNOSIS_TO_FILE_COLUMNS
    )

    df_preprocessing, df_missing = create_preprocessing_dataframe(
        df=df,
        existing_file_columns=existing_file_columns,
        ecg_path=str(data_dir),
    )

    if df_preprocessing.empty:
        return {"error": "No valid ECG files found for preprocessing"}

    preprocessed_df = AnalysisPipeline.save_and_preprocess_data(
        df=df_preprocessing,
        output_folder=str(out_dir),
        preprocessing_folder=str(pre_dir),
        preprocessing_n_workers=1,
    )

    if preprocessed_df is None:
        return {"error": "Preprocessing step failed"}

    results = {}
    errors  = []

    for diagnosis_column in existing_diagnosis_columns:
        ecg_file_column = DIAGNOSIS_TO_FILE_COLUMNS[diagnosis_column]
        df_analysis = create_analysis_dataframe(
            df=df,
            diagnosis_column=diagnosis_column,
            ecg_file_column=ecg_file_column,
            preprocessing_folder=str(pre_dir),
        )

        if df_analysis.empty:
            continue

        signal_model = MODEL_MAPPING[diagnosis_column]['efficientnet']
        bert_model   = MODEL_MAPPING[diagnosis_column]['bert']

        try:
            metrics, df_probabilities = AnalysisPipeline.run_analysis(
                df=df_analysis,
                batch_size=1,
                diagnosis_classifier_device=0,
                signal_processing_device=0,
                signal_processing_model_name=signal_model,
                diagnosis_classifier_model_name=bert_model,
                hugging_face_api_key=hf_token,
            )
            if df_probabilities is not None:
                results[diagnosis_column] = df_probabilities.to_dict(orient="records")
        except Exception as e:
            errors.append(f"{diagnosis_column}: {str(e)}")
            traceback.print_exc()

    return {"status": "ok", "model": "deepecg", "findings": results, "errors": errors}


def handler(job):
    job_input = job["input"]
    try:
        ecg_b64  = job_input.get("ecg_base64", "")
        filename = job_input.get("filename", "ecg.npy")
        suffix   = Path(filename).suffix.lower()

        import numpy as np
        raw      = base64.b64decode(ecg_b64)
        work_dir = Path(tempfile.mkdtemp())
        ecg_path = work_dir / f"ecg{suffix}"
        ecg_path.write_bytes(raw)

        signal = load_signal(ecg_path, suffix)
        if signal is None:
            return {"error": f"Cannot parse file with extension {suffix}"}

        npy_path = work_dir / "ecg.npy"
        np.save(str(npy_path), signal)

        hf_token = os.environ.get("HF_TOKEN", "")
        result   = run_pipeline(npy_path, work_dir, hf_token)
        shutil.rmtree(work_dir, ignore_errors=True)
        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
