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
            import numpy as np
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

    sys.path.insert(0, "/workspace/DeepECG_Docker")
    sys.path.insert(0, "/workspace/DeepECG_Docker/fairseq-signals")
    os.environ["MPLBACKEND"] = "Agg"

    from utils.constants import DIAGNOSIS_TO_FILE_COLUMNS, MODEL_MAPPING
    from utils.analysis_pipeline import AnalysisPipeline
    from main import create_preprocessing_dataframe, create_analysis_dataframe

    data_dir = work_dir / "data"
    out_dir  = work_dir / "out"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    shutil.copy(str(npy_path), str(data_dir / "ecg.npy"))

    sig = np.load(str(npy_path))
    print(f"[DeepECG] signal shape: {sig.shape}, dtype: {sig.dtype}")

    manifest = work_dir / "manifest.csv"
    pd.DataFrame([{
        "patient_id":               "pt001",
        "record_name":              "ecg",
        "data_path":                str(data_dir),
        "ecg_machine_diagnosis":    "",
        "77_classes_ecg_file_name": "",
        "afib_5y":                  "",
        "afib_ecg_file_name":       "",
        "lvef_40":                  "",
        "lvef_40_ecg_file_name":    "",
        "lvef_50":                  "",
        "lvef_50_ecg_file_name":    "",
    }]).to_csv(manifest, index=False)

    pre_df  = create_preprocessing_dataframe(
        str(manifest),
        list(DIAGNOSIS_TO_FILE_COLUMNS.keys()),
        str(data_dir)
    )
    anal_df = create_analysis_dataframe(str(manifest))

    pipeline = AnalysisPipeline(
        preprocessing_df=pre_df,
        analysis_df=anal_df,
        output_dir=str(out_dir),
        data_dir=str(data_dir),
    )
    pipeline.run_analysis()

    findings = {}
    errors   = []
    for f in out_dir.rglob("*.csv"):
        try:
            findings[f.stem] = pd.read_csv(f).to_dict(orient="records")
        except Exception as e:
            errors.append(str(e))

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
