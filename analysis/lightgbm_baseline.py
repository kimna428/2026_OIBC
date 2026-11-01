"""Fixed-round global LightGBM and train-only climatology, Task 1 evaluation."""
from pathlib import Path
import argparse, gc, json, time, resource, sys, importlib.metadata, os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/oibc-lgb-mpl")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from .validation import build_windows, validate_composition, aggregate_site_predictions, score_groups, evaluate_sites, local_hours, TRAIN_END
    from .first_eda import sha
except ImportError:
    from validation import build_windows, validate_composition, aggregate_site_predictions, score_groups, evaluate_sites, local_hours, TRAIN_END
    from first_eda import sha
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'handover 2/01_participant_dataset'
DATASETS=ROOT/'analysis/datasets'
OUT=ROOT/'analysis/results/lightgbm_baseline'
PARAMS=dict(objective='regression',n_estimators=300,learning_rate=.05,num_leaves=31,max_depth=-1,min_child_samples=100,colsample_bytree=1.,subsample=1.,reg_lambda=1.,random_state=20261030,n_jobs=4,deterministic=True,force_col_wise=True,verbosity=-1,device_type='cpu')

def postprocess(cf,capacity,clip_capacity=False):
    cf=np.maximum(np.asarray(cf,dtype=float),0)
    if clip_capacity:cf=np.minimum(cf,1.)
    return cf,cf*np.asarray(capacity)

class Climatology:
    """Hierarchical mean CF; >=24 observations required in site/month/hour cell."""
    def __init__(self,min_count=24):self.min_count=min_count
    def fit(self,d):
        if (local_hours(d.ts)>TRAIN_END).any():raise ValueError('statistics may only fit Train')
        d=d.loc[np.isfinite(d.target_cf)]
        if d.empty:raise ValueError('no finite training targets')
        self.site=d.groupby(['site_id','month','operating_hour'],observed=True).target_cf.agg(['mean','count'])
        self.global_mh=d.groupby(['month','operating_hour'],observed=True).target_cf.mean()
        self.global_h=d.groupby('operating_hour',observed=True).target_cf.mean()
        self.mean=float(d.target_cf.mean())
        return self
    def predict(self,d):
        keys=pd.MultiIndex.from_frame(d[['site_id','month','operating_hour']])
        s=self.site.reindex(keys)
        pred=s['mean'].to_numpy().copy();source=np.repeat('site_month_hour',len(d)).astype(object)
        bad=(s['count'].to_numpy()<self.min_count)|~np.isfinite(pred)
        mh=self.global_mh.reindex(pd.MultiIndex.from_frame(d[['month','operating_hour']])).to_numpy()
        pred[bad]=mh[bad];source[bad]='global_month_hour'
        bad=~np.isfinite(pred);h=d.operating_hour.map(self.global_h).to_numpy()
        pred[bad]=h[bad];source[bad]='global_hour'
        bad=~np.isfinite(pred);pred[bad]=self.mean;source[bad]='global_mean'
        return pred,source
    def save(self,path):
        path.mkdir(exist_ok=True)
        self.site.to_csv(path/'climatology_site.csv');self.global_mh.to_csv(path/'climatology_global_month_hour.csv');self.global_h.to_csv(path/'climatology_global_hour.csv')
        (path/'climatology.json').write_text(json.dumps({'mean':self.mean,'min_count':self.min_count}))

def metrics(d):
    error=d.pred_kwh-d.actual_kwh;cf=error/d.capacity_kw
    weights=d.smp*d.k
    peak=d.ts.dt.hour.between(11,15)
    return dict(rows=len(d),mae_kwh=float(error.abs().mean()),rmse_kwh=float(np.sqrt((error**2).mean())),cf_mae=float(cf.abs().mean()),cf_rmse=float(np.sqrt((cf**2).mean())),weighted_error_krw=float((weights*error.abs()).sum()),weighted_mae_kwh=float((weights*error.abs()).sum()/weights.sum()) if weights.sum() else np.nan,peak_rows=int(peak.sum()),peak_weighted_error_krw=float((weights[peak]*error[peak].abs()).sum()),peak_weighted_mae_kwh=float((weights[peak]*error[peak].abs()).sum()/weights[peak].sum()) if weights[peak].sum() else np.nan)

def prepare_features(d,cols,categories):
    x=d[cols].copy()
    x['site_id']=pd.Categorical(x.site_id,categories=categories)
    for c in cols:
        if x[c].dtype=='float64':x[c]=x[c].astype('float32')
    if not list(x.columns)==cols:raise ValueError('feature order mismatch')
    return x

def md(d):
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))

def plots(preds,importance):
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for col,split in enumerate(('dev','holdout')):
        for model in ('lightgbm','climatology'):
            d=preds[(split,model)]
            daily=d.assign(day=(d.ts-pd.Timedelta(nanoseconds=1)).dt.normalize()).groupby('day')[['actual_kwh','pred_kwh']].sum()/1000
            if model=='lightgbm':axes[0,col].plot(daily.index,daily.actual_kwh,label='Actual',c='black',lw=1)
            axes[0,col].plot(daily.index,daily.pred_kwh,label=model,lw=.8)
        d=preds[(split,'lightgbm')]
        axes[1,col].hexbin(d.actual_kwh,d.pred_kwh,gridsize=60,bins='log',mincnt=1)
        mx=max(d.actual_kwh.max(),d.pred_kwh.max());axes[1,col].plot([0,mx],[0,mx],'r--',lw=.7)
        axes[0,col].set(title=split,ylabel='Daily fleet energy (MWh)');axes[0,col].legend(fontsize=8)
        axes[1,col].set(xlabel='Actual site kWh',ylabel='LightGBM predicted site kWh')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/'actual_vs_predicted.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,8));importance.head(25).sort_values('gain').plot.barh(x='feature',y='gain',ax=ax,legend=False);ax.set_title('LightGBM training gain importance');fig.tight_layout();fig.savefig(OUT/'feature_importance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for split in ('dev','holdout'):
        for model in ('lightgbm','climatology'):
            d=preds[(split,model)].copy();d['hour']=d.ts.dt.hour;d['ae']=(d.pred_kwh-d.actual_kwh).abs()
            axes[0].plot(d.groupby('hour').ae.mean(),label=split+' '+model)
            axes[1].plot(d.groupby(d.operating_day.dt.strftime('%Y-%m')).ae.mean(),marker='o',label=split+' '+model)
    axes[0].set(xlabel='KST ts end hour',ylabel='Site MAE (kWh)');axes[1].set(xlabel='Operating month',ylabel='Site MAE (kWh)');axes[0].legend(fontsize=7);axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'errors_by_time.png',dpi=150);plt.close(fig)

def probe_gpu(lgb):
    result={}
    x=np.arange(80,dtype=float).reshape(40,2);y=np.arange(40,dtype=float)
    for device in ('gpu','cuda'):
        try:
            lgb.LGBMRegressor(device_type=device,n_estimators=1,num_leaves=2,min_child_samples=2,verbosity=-1,n_jobs=1).fit(x,y)
            result[device]={'available':True}
        except Exception as exc:
            result[device]={'available':False,'error':str(exc).strip()}
    (OUT/'gpu_check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clip-capacity',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    import lightgbm as lgb
    probe_gpu(lgb)
    manifest=json.loads((DATASETS/'manifest.json').read_text());cols=manifest['feature_columns']
    assert not set(cols)&set(manifest['audit_columns'])
    # Verify datasets and original inputs against pipeline fingerprints before fitting.
    for n in ('train','dev','holdout'):assert sha(DATASETS/f'{n}.parquet')==manifest['output_files'][n]['sha256']
    for p,h in manifest['input_sha256'].items():assert sha(ROOT/p)==h
    sites=pd.read_csv(DATA/'sites.csv');categories=sorted(sites.site_id)
    comp=sites[['site_id']].assign(group=sites.current_group.map({'A':'G1','B':'G2'}));caps=validate_composition(comp,sites);comp.to_csv(OUT/'composition.csv',index=False)
    # No validation frame is loaded until both models are fit. No eval_set or early stopping.
    train=pd.read_parquet(DATASETS/'train.parquet',columns=list(dict.fromkeys(cols+['ts','target_cf'])),filters=[('target_present','==',True)])
    assert (local_hours(train.ts)<=TRAIN_END).all() and np.isfinite(train.target_cf).all()
    print(f'Train rows {len(train):,}; fitting train-only climatology',flush=True)
    clim=Climatology().fit(train);clim.save(OUT)
    train_rows=len(train);train_start=str(train.ts.min());train_end=str(train.ts.max())
    x=prepare_features(train,cols,categories);y=train.target_cf.to_numpy(copy=True)
    del train;gc.collect()
    model=lgb.LGBMRegressor(**PARAMS)
    fit_start=time.perf_counter();model.fit(x,y,categorical_feature=['site_id']);fit_seconds=time.perf_counter()-fit_start
    model.booster_.save_model(str(OUT/'model.txt'))
    assert model.booster_.feature_name()==cols
    importance=pd.DataFrame({'feature':cols,'gain':model.booster_.feature_importance('gain'),'split_count':model.booster_.feature_importance('split')}).sort_values('gain',ascending=False)
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    sample=x.iloc[:1000];reload=lgb.Booster(model_file=str(OUT/'model.txt'))
    np.testing.assert_allclose(model.predict(sample),reload.predict(sample),rtol=1e-12,atol=1e-12)
    del x,y,sample,reload;gc.collect();print(f'Model fit completed in {fit_seconds:.1f}s',flush=True)
    generation=pd.read_parquet(DATA/'train/generation.parquet');prices=pd.read_csv(DATA/'train/smp.csv')
    windows=build_windows(generation,prices,sites);del generation;gc.collect()
    validation_manifest=json.loads((ROOT/'reports/validation/manifest.json').read_text())
    for p,h in validation_manifest['input_sha256'].items():assert sha(ROOT/p)==h
    results=[];preds={};fallback=[];checks=[]
    for split in ('dev','holdout'):
        d=pd.read_parquet(DATASETS/f'{split}.parquet')
        window=windows[split]
        saved=local_hours(pd.to_datetime(pd.read_csv(ROOT/f'reports/validation/{split}_settlement_times.csv').ts)).as_unit('ns')
        assert window.times.as_unit('ns').equals(saved)
        assert d.target_present.all() and d.site_id.nunique()==253
        x=prepare_features(d,cols,categories)
        lgb_cf=model.predict(x);del x
        clim_cf,source=clim.predict(d)
        for name,n in pd.Series(source).value_counts().items():fallback.append(dict(split=split,source=name,rows=int(n)))
        for name,raw_cf in [('lightgbm',lgb_cf),('climatology',clim_cf)]:
            cf,kwh=postprocess(raw_cf,d.capacity_kw,args.clip_capacity)
            pred=d[['site_id','ts','operating_day','energy','capacity_kw','smp','k']].rename(columns={'energy':'actual_kwh'}).copy()
            pred['pred_cf_raw']=raw_cf;pred['pred_cf']=cf;pred['pred_kwh']=kwh
            if name=='climatology':pred['fallback_source']=source
            assert np.isfinite(kwh).all() and (kwh>=0).all()
            pred.to_csv(OUT/f'{split}_{name}_predictions.csv',index=False)
            site_metrics,site_total=evaluate_sites(pred,window)
            site_metrics.to_csv(OUT/f'{split}_{name}_site_metrics.csv')
            enriched=[]
            for key in ('site_id','hour','month'):
                groups=pred.groupby(key if key=='site_id' else (pred.ts.dt.hour if key=='hour' else pred.operating_day.dt.strftime('%Y-%m')),observed=True)
                records=[{key:g,**metrics(v)} for g,v in groups]
                pd.DataFrame(records).to_csv(OUT/f'{split}_{name}_{key}_metrics.csv',index=False)
            group_pred=aggregate_site_predictions(pred,comp,sites,window)
            group_metrics,total,ledger=score_groups(group_pred,comp,sites,window)
            group_metrics.to_csv(OUT/f'{split}_{name}_group_metrics.csv');ledger.to_csv(OUT/f'{split}_{name}_settlement_ledger.csv',index=False)
            summary=metrics(pred)
            np.testing.assert_allclose([summary['mae_kwh'],summary['rmse_kwh']],[site_total['mae_kwh'],site_total['rmse_kwh']],rtol=1e-12)
            summary.update({k:total[k] for k in ('revenue_krw','penalty_krw','score_krw','used_hours','excluded_hours')})
            results.append(dict(split=split,model=name,**summary))
            checks.append(dict(split=split,model=name,negative_raw_predictions=int((raw_cf<0).sum()),cf_above_one_raw=int((raw_cf>1).sum()),reconstruction_verified=bool(np.allclose(cf*d.capacity_kw,kwh))))
            preds[(split,name)]=pred
        print(f'{split} evaluated on {len(window.times)} fixed complete hours',flush=True)
    comparison=pd.DataFrame(results);comparison.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(fallback).to_csv(OUT/'climatology_fallback.csv',index=False)
    plots(preds,importance)
    for split in ('dev','holdout'):
        r=comparison[comparison.split==split];np.testing.assert_allclose(r.revenue_krw.iloc[0],r.revenue_krw.iloc[1],rtol=1e-12)
    for p,h in manifest['input_sha256'].items():assert sha(ROOT/p)==h
    versions={p:importlib.metadata.version(p) for p in ('lightgbm','numpy','pandas','pyarrow','scikit-learn')}
    run=dict(params=PARAMS,feature_columns=cols,categorical_features=['site_id'],site_categories=categories,clip_capacity=args.clip_capacity,negative_prediction_floor=0.,train_rows=train_rows,train_start=train_start,train_end=train_end,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-started,versions=versions,group_capacity_kw=caps.to_dict(),checks=checks,input_sha256=manifest['input_sha256'],dataset_sha256={n:manifest['output_files'][n]['sha256'] for n in ('train','dev','holdout')},model_sha256=sha(OUT/'model.txt'),gpu_check=json.loads((OUT/'gpu_check.json').read_text()) if (OUT/'gpu_check.json').exists() else {'status':'not probed'},reloaded_model_predictions_match=True)
    (OUT/'run_manifest.json').write_text(json.dumps(run,indent=2)+'\n')
    view=comparison[['split','model','mae_kwh','rmse_kwh','cf_mae','cf_rmse','revenue_krw','penalty_krw','score_krw']]
    findings=[]
    for split_name in ('dev','holdout'):
        t=comparison[comparison.split==split_name].set_index('model')
        findings.append(f"{split_name}: 통계 모델 대비 LightGBM MAE {1-t.loc['lightgbm','mae_kwh']/t.loc['climatology','mae_kwh']:.2%} 감소, 집합 정산 페널티 {1-t.loc['lightgbm','penalty_krw']/t.loc['climatology','penalty_krw']:.2%} 감소.")
    findings_text='\n\n'.join(findings)
    report=f'''# OIBC LightGBM 통합 발전량 베이스라인

## 실행 결과

253개 발전소를 하나의 LightGBM Regressor로 학습했다. 기존 작업 2의 round=17 예보 77개 피처와 작업 1 평가 함수를 사용했다. 실제 실행 결과:

{md(view)}

에너지 오차는 개별 발전소 관측 행을 pooled한 kWh MAE/RMSE다. CF 오차는 각 행의 `(pred_kwh−actual_kwh)/capacity_kw`로 계산한 무차원 MAE/RMSE이며 CF MAE×100은 설비용량 대비 퍼센트포인트다. 정산은 고정 집합 G1/G2의 점수를 합산한 원(KRW)이다. [전체 수치와 11~15시 가중 지표](../analysis/results/lightgbm_baseline/comparison.csv).

{findings_text}

두 구간 모두 원시 CF>1 LightGBM 예측은 0건이었다. 상한 clipping은 적용하지 않았다. 실제 0 근처에서도 큰 양의 예측 잔차가 남는 사례가 있어, 이후 주간 무발전 사례와 발전소별 오차를 별도 검토할 필요가 있다.

## 학습·재현 설정

Train은 ts≤2025-01-01 00:00 KST, Dev는 2025-01-01 01:00~2025-04-01 00:00, Holdout은 2025-04-01 01:00~2025-06-01 00:00이다. 우측닫힘 운영일 경계를 변경하지 않았다. 유한 실제 target {train_rows:,}개만 학습했고, 관측 기간은 {train_start} ~ {train_end}이다.

고정 300 estimators, learning_rate=0.05, num_leaves=31, min_child_samples=100, L2=1, seed=20261030, CPU 4 threads, deterministic/force_col_wise를 사용했다. Dev/Holdout 모두 fit의 eval_set에 전달하지 않았으며 조기 종료·파라미터 탐색·Holdout 기반 선택을 하지 않았다. Train으로만 학습한 모델을 그대로 두 구간에 평가했다. 모델 학습 {fit_seconds:.2f}초, 전체 실행 {run['total_seconds']:.2f}초.

설치된 LightGBM은 {versions['lightgbm']}이다. 최초 import는 libomp 부재로 실패했으며 Homebrew libomp 설치 후 실행했다. GPU 확인 결과는 [gpu_check.json](../analysis/results/lightgbm_baseline/gpu_check.json)에 기록했다. CPU를 사용한다.

[모델](../analysis/results/lightgbm_baseline/model.txt), [정확한 파라미터·피처 순서·범주·버전·입력 해시](../analysis/results/lightgbm_baseline/run_manifest.json). 저장한 모델을 다시 로드하여 1,000개 학습 입력의 예측이 일치함을 확인했다. 새 추론에도 동일 feature_columns 순서와 site_categories를 적용한다.

## 피처·target·후처리

site_id는 253개 고정 범주형이다. capacity, 위경도, 기상 지점까지 거리, 운영일 달력/주기, WA/WB 예보와 결측 플래그를 사용했다. 운영일 시각은 1~24, 월/dayofyear는 `(ts−1ns)` 날짜 기준이다. 실측 기상·lag/rolling·외부 데이터·target 통계 피처를 사용하지 않는다. SMP/k/실제 발전량/target 유무/target 이상치 플래그는 모델 입력에서 제외했다. 입력 float64는 모델에 전달할 때만 float32로 변환하며 원본·Parquet은 변경하지 않는다. 예보 NaN은 LightGBM native missing 처리에 맡긴다.

목표는 energy/capacity_kw이다. 음수 target 및 CF>1 관측을 유지했다. 예측 CF를 0 이상으로 제한하고 capacity_kw를 곱해 kWh로 복원했다. **용량 상한 clipping={args.clip_capacity}**이며 기본값은 미적용이다. `--clip-capacity`로 상한 CF=1을 선택할 수 있으나 이번 비교에서는 설정을 변경하지 않았다. 별도의 야간 고정 후처리는 없다. 따라서 낮은 야간 양의 예측이 남을 수 있다.

## Train 통계 비교 모델

발전소×운영일 월×운영일 hour의 Train 평균 CF를 사용한다. 셀 표본이 24개 미만이면 전체 발전소의 Train 월×hour 평균, 없으면 Train hour 평균, 다시 없으면 Train 전체 평균으로 대체한다. fit에서 Train 경계 이후 데이터가 들어오면 실패한다. Dev/Holdout target으로 통계를 갱신하지 않는다. [셀별 통계](../analysis/results/lightgbm_baseline/climatology_site.csv), [구간별 fallback 사용량](../analysis/results/lightgbm_baseline/climatology_fallback.csv). 동일 음수/상한 후처리를 적용한다.

## 세부 성능과 정산

시간대별 오차는 KST **ts 끝 시각**, 월별 오차는 **운영일 월**로 집계한다. `*_hour_metrics.csv`, `*_month_metrics.csv`, `*_site_id_metrics.csv`에 MAE/RMSE/CF 오차/가중 오차가 있다. `*_site_metrics.csv`는 작업 1의 개별 평가 함수 결과다.

`weighted_error_krw=Σ SMP×k×|site error|`, `peak_weighted_error_krw`는 그중 KST ts 11~15시만 합산한 값이다. 각각 weighted MAE는 대응 Σ(SMP×k)로 나눈 kWh다. 이 **개별 발전소 가중 오차**는 집합 내 오차 상쇄가 없으므로 집합 정산 페널티와 다르다. 집합 정산은 작업 1의 `aggregate_site_predictions`와 `score_groups`로만 계산한다.

기준 배정은 sites.current_group의 A→G1, B→G2이며 용량은 {caps.to_dict()} kW다. 두 그룹이 전체 253개를 정확히 분할하고 각각 20,000 kW 이상인지 검증했다. 두 모델은 같은 배정과 고정 정산 시각을 사용한다. Dev 2,160시간, Holdout 1,464시간 모두 실제값 완전으로 제외 0시간이다. 실제 그룹 발전량에 의존하는 매전수익이 두 모델에서 같음을 검증했다. 그룹별 금액과 매 시각 ledger는 `*_group_metrics.csv`, `*_settlement_ledger.csv`에 남겼다.

## 예측 파일과 그래프

[Dev LightGBM 예측](../analysis/results/lightgbm_baseline/dev_lightgbm_predictions.csv), [Holdout LightGBM 예측](../analysis/results/lightgbm_baseline/holdout_lightgbm_predictions.csv). 각각 site_id, ts(KST), actual_kwh, pred_kwh, capacity_kw와 원시/후처리 CF를 포함한다. 비교 모델 예측도 같은 폴더에 있다. 모든 예측/실제는 원래 kWh 단위다.

![실제·예측 비교](../analysis/results/lightgbm_baseline/actual_vs_predicted.png)

위 그래프는 전체 발전소의 완전한 일별 합계이며 아래는 개별 시간 관측의 밀도도다.

![시간·월별 MAE](../analysis/results/lightgbm_baseline/errors_by_time.png)

![학습 gain 중요도](../analysis/results/lightgbm_baseline/feature_importance.png)

[전체 feature importance](../analysis/results/lightgbm_baseline/feature_importance.csv)는 gain과 split count를 포함한다. 상관된 기상 변수 사이 중요도가 분산될 수 있고 gain은 인과 효과가 아니다.

## 실행 및 검증

```sh
uv pip install --python .venv/bin/python -r analysis/lightgbm_requirements.txt
# macOS에서 libomp가 없는 경우: brew install libomp
MPLCONFIGDIR=/tmp/oibc-lgb-mpl .venv/bin/python analysis/lightgbm_baseline.py
.venv/bin/python -m unittest discover -s analysis -p test_lightgbm_baseline.py -v
.venv/bin/python analysis/audit_lightgbm_baseline.py
```

기존 데이터셋/원본 해시를 학습 전에 검증했고 원본은 실행 후에도 해시가 같다. target 관측만 학습, Train 경계, feature/audit 분리, 253개소, 고정 정산 시각, 예측 유한·비음수, kWh 복원, 평가 함수와 MAE/RMSE 일치, 모델 재로드 예측 일치를 확인했다. 테스트 출력은 [test_results.txt](../analysis/results/lightgbm_baseline/test_results.txt)에 기록했다. 신규 단위 테스트 4개와 기존 평가 테스트 11개가 통과했다. 독립 감사는 원본 발전량 전수 대조, 역사적 SMP와 k, 정산 내역 재계산, 저장 모델로 두 구간의 표본 예측 재현을 확인하며 [verification.json](../analysis/results/lightgbm_baseline/verification.json)에 결과를 기록한다. Test 예측 생성이나 Train+Dev 재학습은 이번 단계에 포함하지 않는다.
'''
    (ROOT/'reports/baseline_lightgbm.md').write_text(report)
    print(view.to_string(index=False),flush=True)
if __name__=='__main__':main()
