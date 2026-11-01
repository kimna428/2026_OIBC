"""Streaming site diagnostics and the repository's official fixed-completeness scorer."""
import numpy as np
import pandas as pd
from analysis.validation import EvaluationWindow,aggregate_site_predictions,score_groups,hourly_weights,validate_composition
from .features import day_hours

class Evaluator:
    def __init__(self,cache,seen_hours):
        self.cache=cache; self.seen=np.asarray(seen_hours); self.rows=[]; self.group_rows=[]; self.ledger=[]; self.incomplete_hours=0; self.complete_hours=0
        validate_composition(cache.composition,cache.sites)
    def add_day(self,day_index,pred_cf):
        c=self.cache; pred=np.maximum(np.asarray(pred_cf,dtype=float),0)
        if pred.shape!=(c.num_sites,24) or not np.isfinite(pred).all(): raise ValueError('Prediction grid must contain all sites and 24 finite values')
        actual_cf=np.asarray(c.target[day_index]); actual=np.asarray(c.energy[day_index]); mask=np.isfinite(actual_cf)
        capacity=c.sites.capacity_kw.to_numpy(); times=day_hours(c.days[day_index]); solar=np.asarray(c.temporal[day_index])[...,c.manifest['temporal_features'].index('solar_elevation')]
        err_cf=np.abs(pred-actual_cf); err_kwh=np.abs(pred*capacity[:,None]-actual)
        season=['winter','spring','summer','autumn'][(c.days[day_index].month%12)//3]
        for key,values in [('site_id',c.sites.site_id.to_numpy()),('cohort',np.where(self.seen>=2160,'existing_90d','new_under90d'))]:
            for label in np.unique(values):
                selected=mask&(values==label)[:,None]
                self.rows.append((key,str(label),int(selected.sum()),float(err_cf[selected].sum()),float(err_kwh[selected].sum())))
        for label,selected in [('all',mask),('day',mask&(solar>0)),('night',mask&(solar<=0)),(season,mask)]:
            dimension='overall' if label=='all' else ('season' if label==season else 'period')
            self.rows.append((dimension,label,int(selected.sum()),float(err_cf[selected].sum()),float(err_kwh[selected].sum())))
        complete=mask.all(axis=0); self.complete_hours+=int(complete.sum()); self.incomplete_hours+=int((~complete).sum())
        if not complete.any(): return
        wide=pd.DataFrame(actual.T,index=times,columns=c.sites.site_id)
        detail=pd.DataFrame({'included':complete},index=times)
        window=EvaluationWindow('deep_learning',wide,pd.Series(np.asarray(c.smp[day_index]),index=times),detail)
        p=pd.DataFrame({'site_id':np.repeat(c.sites.site_id.to_numpy(),24),'ts':np.tile(times,c.num_sites),'pred_kwh':(pred*capacity[:,None]).ravel()})
        group_pred=aggregate_site_predictions(p,c.composition,c.sites,window)
        _,_,ledger=score_groups(group_pred,c.composition,c.sites,window)
        self.ledger.append(ledger)
    def finish(self):
        d=pd.DataFrame(self.rows,columns=['dimension','label','rows','abs_cf_sum','abs_kwh_sum'])
        d=d.groupby(['dimension','label'],as_index=False)[['rows','abs_cf_sum','abs_kwh_sum']].sum()
        d['cf_mae']=d.abs_cf_sum/d.rows.replace(0,np.nan); d['mae_kwh']=d.abs_kwh_sum/d.rows.replace(0,np.nan)
        overall=d[d.dimension=='overall'].iloc[0]
        result={'site_cf_mae':float(overall.cf_mae) if overall.rows else None,'site_kwh_mae':float(overall.mae_kwh) if overall.rows else None,'observed_site_hours':int(overall.rows),'scored_hours':self.complete_hours,'excluded_hours':self.incomplete_hours,'score_available':bool(self.complete_hours)}
        groups=[]
        if self.ledger:
            ledger=pd.concat(self.ledger,ignore_index=True); ledger['abs_error_kwh']=(ledger.pred_kwh-ledger.actual_kwh).abs()
            for group,z in [('total',ledger)]+list(ledger.groupby('group')):
                groups.append({'group':group,'mae_kwh':float(z.abs_error_kwh.mean()),'penalty_krw':float(z.penalty_krw.sum()),'score_krw':float(z.score_krw.sum()),'scored_hours':self.complete_hours})
            result.update(groups[0]); result['penalty_per_hour']=result['penalty_krw']/self.complete_hours
        else:
            ledger=pd.DataFrame(); result.update({'mae_kwh':None,'penalty_krw':None,'score_krw':None,'penalty_per_hour':None})
        return result,d,pd.DataFrame(groups),ledger
