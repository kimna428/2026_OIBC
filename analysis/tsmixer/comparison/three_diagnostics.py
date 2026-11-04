"""Per-site/per-group complementarity, with descriptive half-blend penalties."""
import itertools
import numpy as np
import pandas as pd
from .diagnostics import correlation


def complementarity(grid,predictions,ledgers,output):
    residual={name:grid.validate_prediction(p)-grid.actual for name,p in predictions.items()}
    site_rows=[]
    for a,b in itertools.combinations(residual,2):
        for s,sid in enumerate(grid.sites.site_id):
            observed=grid.observed[:,s]
            ra,rb=residual[a][:,s][observed],residual[b][:,s][observed]
            site_rows.append(dict(model_a=a,model_b=b,site_id=sid,observed_rows=len(ra),
                       residual_correlation=correlation(ra,rb),
                       opposite_sign_fraction=float(np.mean(ra*rb<0)) if len(ra) else None))
    pd.DataFrame(site_rows,columns=['model_a','model_b','site_id','observed_rows','residual_correlation','opposite_sign_fraction']).to_csv(output/'site_residual_correlations.csv',index=False)
    group_rows=[]
    for assignment,models in ledgers.items():
        for (a,la),(b,lb) in itertools.combinations(models.items(),2):
            joined=la.merge(lb,on=['group','ts'],suffixes=('_a','_b'),validate='one_to_one')
            if len(joined)!=len(la) or len(joined)!=len(lb):
                raise ValueError('Complementarity requires identical group/timestamp masks')
            for group,d in joined.groupby('group'):
                ra,rb=d.residual_kwh_a.to_numpy(),d.residual_kwh_b.to_numpy()
                half=np.abs((ra+rb)/2)
                group_rows.append(dict(assignment=assignment,group=group,model_a=a,model_b=b,hours=len(d),
                        residual_correlation=correlation(ra,rb),opposite_sign_fraction=float(np.mean(ra*rb<0)),
                        W_a=float(d.penalty_krw_a.sum()),W_b=float(d.penalty_krw_b.sum()),
                        W_half_exploratory=float((.1*d.smp_a*d.k_a*half).sum()),
                        independent_validation=False))
    pd.DataFrame(group_rows,columns=['assignment','group','model_a','model_b','hours','residual_correlation',
                    'opposite_sign_fraction','W_a','W_b','W_half_exploratory','independent_validation']).to_csv(output/'group_complementarity.csv',index=False)
    return dict(site_pair_rows=len(site_rows),group_pair_rows=len(group_rows),interpretation='descriptive only; low correlation alone is not a recommendation')
