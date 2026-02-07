"""Gradio app for the MirAI 3-stage inference pipeline.

Run: python gradio_app.py
Open http://localhost:7860

This script loads the stage artifacts once and serves a small web UI to
run Stage1 -> Stage2 -> Stage3 inference on user-provided inputs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from xgboost import XGBClassifier


class MirAIEngine:
    def __init__(self, base_dir=None):
        base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
        # allow nested repo structure where models live one level up
        if not (base / 'stage 1').exists() and (base.parent / 'stage 1').exists():
            base = base.parent

        self.base = base
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        # load artifacts for stages
        self._load_stage('s1', 'stage 1', 'stage1_model.json', 'stage1_imputer.pkl', 'stage1_scaler.pkl')
        self._load_stage('s2', 'stage 2', 'stage2_model.json', 'stage2_imputer.pkl', 'stage2_scaler.pkl')
        self._load_stage('s3', 'stage 3', 'stage3_model.json', 'stage3_imputer.pkl', 'stage3_scaler.pkl')

    def _load_stage(self, key, folder, model_name, imputer_name, scaler_name):
        d = self.base / folder
        try:
            m = XGBClassifier()
            m.load_model(str(d / model_name))
            self.models[key] = m
            self.imputers[key] = joblib.load(str(d / imputer_name))
            self.scalers[key] = joblib.load(str(d / scaler_name))
        except Exception as e:
            raise RuntimeError(f'Failed loading {folder} artifacts from {d}: {e}')

    @staticmethod
    def _encode_ptgender(v):
        if v is None:
            return np.nan
        if isinstance(v, (int, float)):
            return int(v)
        v = str(v).strip().lower()
        if v.startswith('m'):
            return 1
        if v.startswith('f'):
            return 0
        return np.nan

    def infer(self, AGE, PTGENDER, PTEDUCAT, FAQ, EcogPtMem, EcogPtTotal, GENOTYPE, pT217_F, AB42_F, AB40_F, NfL_Q, threshold=0.5):
        # Stage1
        s1_feats = ['AGE', 'PTGENDER', 'PTEDUCAT', 'FAQ', 'EcogPtMem', 'EcogPtTotal']
        s1_row = {
            'AGE': AGE,
            'PTGENDER': self._encode_ptgender(PTGENDER),
            'PTEDUCAT': PTEDUCAT,
            'FAQ': FAQ,
            'EcogPtMem': EcogPtMem,
            'EcogPtTotal': EcogPtTotal
        }
        df1 = pd.DataFrame([s1_row])
        try:
            X1_imp = self.imputers['s1'].transform(df1)
            X1_scaled = self.scalers['s1'].transform(X1_imp)
            p1 = float(self.models['s1'].predict_proba(X1_scaled)[0, 1])
        except Exception as e:
            return {'error': f'Stage1 error: {e}'}, None

        # APOE4 count
        try:
            apoe = int(str(GENOTYPE).count('4')) if GENOTYPE else 0
        except Exception:
            apoe = 0

        # Stage2
        df2 = pd.DataFrame([[p1, apoe]], columns=['Stage1_Prob', 'APOE4_Count'])
        try:
            X2_imp = self.imputers['s2'].transform(df2)
            X2_scaled = self.scalers['s2'].transform(X2_imp)
            p2 = float(self.models['s2'].predict_proba(X2_scaled)[0, 1])
        except Exception as e:
            return {'error': f'Stage2 error: {e}'}, None

        # Stage3
        plasma_keys = ['pT217_F', 'AB42_F', 'AB40_F', 'NfL_Q']
        plasma_vals = [pT217_F, AB42_F, AB40_F, NfL_Q]
        df3 = pd.DataFrame([[p2] + plasma_vals], columns=['Stage2_Prob'] + plasma_keys)
        try:
            X3_imp = self.imputers['s3'].transform(df3)
            X3_scaled = self.scalers['s3'].transform(X3_imp)
            p3 = float(self.models['s3'].predict_proba(X3_scaled)[0, 1])
        except Exception as e:
            return {'error': f'Stage3 error: {e}'}, None

        # human-friendly
        summary = {
            'Stage1_Prob': round(p1 * 100, 2),
            'Stage2_Prob': round(p2 * 100, 2),
            'Stage3_Prob': round(p3 * 100, 2),
            'Stage3_Flag': ('POSITIVE' if p3 >= threshold else 'NEGATIVE')
        }
        details = {
            'Stage1_input': s1_row,
            'Stage2_input': {'Stage1_Prob': p1, 'APOE4_Count': apoe},
            'Stage3_input': {'Stage2_Prob': p2, **dict(zip(plasma_keys, plasma_vals))}
        }
        return summary, details


def build_interface(engine: MirAIEngine):
    with gr.Blocks() as demo:
        gr.Markdown('# MirAI — 3-Stage Inference Demo')
        with gr.Row():
            with gr.Column():
                AGE = gr.Number(value=70, label='Age')
                PTGENDER = gr.Dropdown(choices=['Male', 'Female', 'Other'], value='Male', label='Gender')
                PTEDUCAT = gr.Number(value=12, label='Education (years)')
                FAQ = gr.Number(value=0, label='FAQ (functional activities)')
                EcogPtMem = gr.Number(value=0.5, label='EcogPtMem')
                EcogPtTotal = gr.Number(value=1.0, label='EcogPtTotal')
                GENOTYPE = gr.Textbox(value='4/4', label='GENOTYPE (e.g., 3/4, 4/4)')
            with gr.Column():
                pT217_F = gr.Number(value=0.5, label='pT217_F')
                AB42_F = gr.Number(value=1.0, label='AB42_F')
                AB40_F = gr.Number(value=14.0, label='AB40_F')
                NfL_Q = gr.Number(value=10.0, label='NfL_Q')
                threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label='Decision threshold')
                run_btn = gr.Button('Run Inference')

        summary_out = gr.JSON(label='Summary (percent)')
        details_out = gr.JSON(label='Details')

        def _run(AGE, PTGENDER, PTEDUCAT, FAQ, EcogPtMem, EcogPtTotal, GENOTYPE, pT217_F, AB42_F, AB40_F, NfL_Q, threshold):
            summary, details = engine.infer(AGE, PTGENDER, PTEDUCAT, FAQ, EcogPtMem, EcogPtTotal, GENOTYPE, pT217_F, AB42_F, AB40_F, NfL_Q, threshold)
            if isinstance(summary, dict) and 'error' in summary:
                return gr.update(value=summary), {}
            return summary, details

        run_btn.click(_run, inputs=[AGE, PTGENDER, PTEDUCAT, FAQ, EcogPtMem, EcogPtTotal, GENOTYPE, pT217_F, AB42_F, AB40_F, NfL_Q, threshold], outputs=[summary_out, details_out])

    return demo


if __name__ == '__main__':
    try:
        engine = MirAIEngine()
    except Exception as e:
        print('Failed to load artifacts:', e)
        raise
    demo = build_interface(engine)
    demo.launch(server_name='0.0.0.0', share=False)
