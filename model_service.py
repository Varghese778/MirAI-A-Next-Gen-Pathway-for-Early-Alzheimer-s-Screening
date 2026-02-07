# Demo cell: load artifacts and run Stage 1 -> 2 -> 3 with clear outputs
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

class MirAI_Engine_Simple:
    def __init__(self, base_dir):
        base = Path(base_dir)
        self.models = {}
        self.imputers = {}
        self.scalers = {}
        # Stage 1 artifacts
        self.models['s1'] = XGBClassifier()
        self.models['s1'].load_model(str(base / 'stage 1' / 'stage1_model.json'))
        self.imputers['s1'] = joblib.load(str(base / 'stage 1' / 'stage1_imputer.pkl'))
        self.scalers['s1'] = joblib.load(str(base / 'stage 1' / 'stage1_scaler.pkl'))
        # Stage 2 artifacts
        self.models['s2'] = XGBClassifier()
        self.models['s2'].load_model(str(base / 'stage 2' / 'stage2_model.json'))
        self.imputers['s2'] = joblib.load(str(base / 'stage 2' / 'stage2_imputer.pkl'))
        self.scalers['s2'] = joblib.load(str(base / 'stage 2' / 'stage2_scaler.pkl'))
        # Stage 3 artifacts
        self.models['s3'] = XGBClassifier()
        self.models['s3'].load_model(str(base / 'stage 3' / 'stage3_model.json'))
        self.imputers['s3'] = joblib.load(str(base / 'stage 3' / 'stage3_imputer.pkl'))
        self.scalers['s3'] = joblib.load(str(base / 'stage 3' / 'stage3_scaler.pkl'))

    def _encode_ptgender(self, v):
        if isinstance(v, (int, float)): return int(v)
        if not isinstance(v, str): return np.nan
        v = v.strip().lower()
        if v.startswith('m'): return 1
        if v.startswith('f'): return 0
        return np.nan

# determine models base dir
try:
    _base = models_dir
except Exception:
    _base = Path(adnimerge_path).parents[1]
engine_demo = MirAI_Engine_Simple(_base)

# Example patients
high = {'AGE':75,'PTGENDER':'Male','PTEDUCAT':12,'FAQ':10,'EcogPtMem':3.5,'EcogPtTotal':8.0,'GENOTYPE':'4/4','pT217_F':1.8,'AB42_F':0.5,'AB40_F':12.0,'NfL_Q':35.0}
low =  {'AGE':68,'PTGENDER':'Female','PTEDUCAT':16,'FAQ':0,'EcogPtMem':0.5,'EcogPtTotal':1.0,'GENOTYPE':'','pT217_F':0.2,'AB42_F':1.0,'AB40_F':14.0,'NfL_Q':10.0}

def _round_list(a):
    return [float(np.round(x,4)) for x in np.asarray(a).ravel().tolist()]

def run_full_demo(patient, name='Patient', threshold=0.5):
    print('\n==== {} ===='.format(name))
    # Stage 1
    s1_feats = ['AGE','PTGENDER','PTEDUCAT','FAQ','EcogPtMem','EcogPtTotal']
    s1_row = {f: (engine_demo._encode_ptgender(patient.get(f)) if f=='PTGENDER' else patient.get(f, np.nan)) for f in s1_feats}
    df1 = pd.DataFrame([s1_row])
    try:
        X1_imp = engine_demo.imputers['s1'].transform(df1)
        X1_scaled = engine_demo.scalers['s1'].transform(X1_imp)
        p1 = float(engine_demo.models['s1'].predict_proba(X1_scaled)[0,1])
        imp_map = dict(zip(s1_feats, _round_list(X1_imp[0])))
        scaled_map = dict(zip(s1_feats, _round_list(X1_scaled[0])))
        flag1 = 'HIGH' if p1 >= threshold else 'LOW'
        print('Stage1 - original inputs:', df1.to_dict(orient='records'))
        print('Stage1 - imputed (feature: value):', imp_map)
        print('Stage1 - scaled (feature: value):', scaled_map)
        print('Stage1 probability: {:.1f}%   flag: {}'.format(p1*100, flag1))
    except Exception as e:
        print('Stage1 error:', e); p1 = None

    # APOE count
    try:
        apoe = int(str(patient.get('GENOTYPE','')).count('4')) if patient.get('GENOTYPE') else 0
    except Exception:
        apoe = 0
    # Stage 2
    try:
        df2 = pd.DataFrame([[p1 if p1 is not None else 0.0, apoe]], columns=['Stage1_Prob','APOE4_Count'])
        X2_imp = engine_demo.imputers['s2'].transform(df2)
        X2_scaled = engine_demo.scalers['s2'].transform(X2_imp)
        p2 = float(engine_demo.models['s2'].predict_proba(X2_scaled)[0,1])
        imp_map2 = dict(zip(['Stage1_Prob','APOE4_Count'], _round_list(X2_imp[0])))
        scaled_map2 = dict(zip(['Stage1_Prob','APOE4_Count'], _round_list(X2_scaled[0])))
        flag2 = 'HIGH' if p2 >= threshold else 'LOW'
        print('Stage2 - input:', df2.to_dict(orient='records'))
        print('Stage2 - imputed:', imp_map2)
        print('Stage2 - scaled:', scaled_map2)
        print('Stage2 probability: {:.1f}%   flag: {}'.format(p2*100, flag2))
    except Exception as e:
        print('Stage2 error:', e); p2 = None

    # Stage 3
    plasma_keys = ['pT217_F','AB42_F','AB40_F','NfL_Q']
    plasma_vals = [patient.get(k, np.nan) for k in plasma_keys]
    try:
        df3 = pd.DataFrame([[p2 if p2 is not None else 0.0] + plasma_vals], columns=['Stage2_Prob']+plasma_keys)
        X3_imp = engine_demo.imputers['s3'].transform(df3)
        X3_scaled = engine_demo.scalers['s3'].transform(X3_imp)
        p3 = float(engine_demo.models['s3'].predict_proba(X3_scaled)[0,1])
        imp_map3 = dict(zip(['Stage2_Prob']+plasma_keys, _round_list(X3_imp[0])))
        scaled_map3 = dict(zip(['Stage2_Prob']+plasma_keys, _round_list(X3_scaled[0])))
        flag3 = 'POSITIVE' if p3 >= threshold else 'NEGATIVE'
        print('Stage3 - input:', df3.to_dict(orient='records'))
        print('Stage3 - imputed:', imp_map3)
        print('Stage3 - scaled:', scaled_map3)
        print('Stage3 probability: {:.1f}%   flag: {}'.format(p3*100, flag3))
    except Exception as e:
        print('Stage3 error:', e); p3 = None

    # Summary
    print('\nSummary for {}:'.format(name))
    print('  Stage1:', 'N/A' if p1 is None else '{:.1f}%'.format(p1*100))
    print('  Stage2:', 'N/A' if p2 is None else '{:.1f}%'.format(p2*100))
    print('  Stage3:', 'N/A' if (('p3' not in locals()) or p3 is None) else '{:.1f}%'.format(p3*100))
    return {'s1':p1,'s2':p2,'s3':(p3 if ('p3' in locals()) else None)}

# Run demo for both examples
out_high = run_full_demo(high, 'High-risk example')
out_low = run_full_demo(low, 'Low-risk example')
print('\nFinal summary (raw probabilities):', {'high':out_high, 'low':out_low})