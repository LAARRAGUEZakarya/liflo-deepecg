def run_pipeline(npy_path, work_dir):
    import numpy as np
    import pandas as pd
    import sys, os

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

    manifest = work_dir / "manifest.csv"
    pd.DataFrame([{
        "patient_id":  "pt001",
        "record_name": "ecg",
        "data_path":   str(data_dir),
        "AF": "", "IAVB": "", "LBBB": "", "RBBB": ""
    }]).to_csv(manifest, index=False)

    # Inspect the signal shape for debugging
    sig = np.load(str(npy_path))
    print(f"[DeepECG] signal shape: {sig.shape}, dtype: {sig.dtype}, min: {sig.min():.3f}, max: {sig.max():.3f}")

    pre_df  = create_preprocessing_dataframe(str(manifest), list(DIAGNOSIS_TO_FILE_COLUMNS.keys()), str(data_dir))
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
            df = pd.read_csv(f)
            findings[f.stem] = df.to_dict(orient="records")
        except Exception as e:
            errors.append(str(e))

    return {
        "status":   "ok",
        "model":    "deepecg",
        "findings": findings,
        "errors":   errors,
    }
