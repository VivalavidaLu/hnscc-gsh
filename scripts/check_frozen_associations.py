"""Recompute reported correlations and patient LOO from frozen aggregate data.

APIs: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
No raw-expression reconstruction or full figure regeneration is claimed.
"""
from pathlib import Path
import importlib.metadata
import json
import math
import platform
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = pd.read_csv(ROOT/'data/processed/p1_patient_phase_cxcl9_cxcl10.csv')
    data = data.loc[data.n_P1_cells >= 10].copy()
    expected = {'CXCL9_mean_logCP10K':(-0.20336234318926663,0.5051743132421285),
                'CXCL10_mean_logCP10K':(-0.5878007453826749,0.034628795889980746),
                'CXCL10_detection_fraction':(-0.5989438874752373,0.030539122224189395)}
    result = {'python':platform.python_version(),
              'versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy']},
              'n_patient_phase':len(data),'n_patients':int(data.Patient.nunique()),'correlations':{},'loo':{}}
    for outcome,reference in expected.items():
        pair = data[['GSH_cystine_score',outcome]].dropna()
        rho,p = spearmanr(pair.iloc[:,0],pair.iloc[:,1])
        if not (math.isclose(rho,reference[0],abs_tol=1e-10) and math.isclose(p,reference[1],abs_tol=1e-10)):
            raise ValueError('Frozen correlation changed: '+outcome)
        result['correlations'][outcome]={'rho':float(rho),'p':float(p),'n':len(pair)}
    for gene in ['CXCL9','CXCL10']:
        rhos=[]
        for patient in sorted(data.Patient.unique()):
            frame=data.loc[data.Patient.ne(patient)]
            rhos.append(float(spearmanr(frame.GSH_cystine_score,frame[gene+'_mean_logCP10K']).statistic))
        result['loo'][gene]={'negative':sum(v<0 for v in rhos),'total':len(rhos),'range':[min(rhos),max(rhos)]}
    out=ROOT/'results'/'frozen_checks'
    out.mkdir(parents=True,exist_ok=True)
    (out/'associations.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
