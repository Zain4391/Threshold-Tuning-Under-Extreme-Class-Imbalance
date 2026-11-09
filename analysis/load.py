import os, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(kind):  # kind: A,B,C,D or S_A,S_C,S_D
    out = []
    for ds, tag in (("fraud", "fraud"), ("cov", "covtype"), ("prot", "protein_homo")):
        d = pd.read_csv(f"{ROOT}/v2_{kind}_{ds}/raw_{kind}.csv"); d["dataset"] = tag; out.append(d)
    return pd.concat(out, ignore_index=True)
