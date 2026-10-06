#!/bin/bash
# Build ~/nyiso-us/.venv on gene (i5-3570K, no AVX2; GTX 1060 sm_61). Torch from the cu126 index,
# the same approach as ~/dream-house/ComfyUI/.venv (pip cache makes it mostly offline).
# Every import is tested in a fresh process; exit 132 = SIGILL = wheel needs AVX2.
set -u
V=~/nyiso-us/.venv
[ -x $V/bin/python ] || python3 -m venv $V
$V/bin/pip install -q --upgrade pip
$V/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 || echo "TORCH INSTALL FAILED"
$V/bin/pip install pandas pyarrow numpy scikit-learn pytest tzdata || echo "BASE INSTALL FAILED"
$V/bin/pip install lightgbm || echo "LIGHTGBM INSTALL FAILED"
echo "--- import tests"
for mod in numpy pandas pyarrow pyarrow.parquet sklearn sklearn.ensemble torch lightgbm pytest; do
  $V/bin/python -c "import $mod" >/dev/null 2>&1; echo "import $mod exit=$?"
done
echo "--- functional tests"
$V/bin/python -c "import numpy as np; a=np.random.rand(500,500); print('numpy matmul ok', float((a@a).sum())>0)"; echo "numpy-func exit=$?"
$V/bin/python -c "
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, tempfile, os
df=pd.DataFrame({'t':pd.date_range('2020-01-01',periods=10,freq='h',tz='America/New_York'),'x':range(10)})
p=os.path.join(tempfile.mkdtemp(),'a.parquet'); df.to_parquet(p); print('parquet roundtrip ok', pd.read_parquet(p).equals(df))"; echo "parquet-func exit=$?"
$V/bin/python -c "
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
X=np.random.rand(2000,8); y=X[:,0]*3+np.random.rand(2000)
print('sklearn HGB ok', HistGradientBoostingRegressor(max_iter=50).fit(X,y).score(X,y))"; echo "sklearn-func exit=$?"
$V/bin/python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
x=torch.randn(512,512,device='cuda'); print('gpu matmul ok', float((x@x).sum())==float((x@x).sum()), torch.cuda.get_device_name(0))
m=torch.nn.LSTM(8,32,batch_first=True).cuda(); o,_=m(torch.randn(4,24,8,device='cuda')); o.sum().backward(); print('lstm fwd/bwd ok')"; echo "torch-func exit=$?"
$V/bin/python -c "
import numpy as np, lightgbm as lgb
X=np.random.rand(2000,8); y=X[:,0]*3+np.random.rand(2000)
m=lgb.LGBMRegressor(n_estimators=50,verbose=-1).fit(X,y); print('lightgbm ok', lgb.__version__, m.score(X,y) if hasattr(m,'score') else '')"; echo "lightgbm-func exit=$?"
$V/bin/pip freeze > ~/nyiso-us/logs/venv-freeze.txt
echo "SETUP DONE"
