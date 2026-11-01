"""One global round-17 CatBoost CF forecaster, with a chronological holdout.

Only observed generation rows become targets. No observed-weather input,
generation imputation, target clipping, validation tuning or group optimization.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import html
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import catboost
from catboost import CatBoostRegressor, Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'handover 2/01_participant_dataset'
DEFAULT_OUT = ROOT/'analysis/results/forecast_baseline'
CATEGORICAL = ['site_id', 'wa_station', 'wb_station']
METADATA = CATEGORICAL + ['capacity_kw','wa_dist','wb_dist','lat','lon']
TIME_FEATURES = ['hour','month','dayofyear','hour_sin','hour_cos','dayofyear_sin','dayofyear_cos']
REPORT_PLOTS = [('hourly_profile','시간별 CF 예측 및 오차'),('site_errors','발전소별 MAE 상위 20개소'),
    ('residual_calibration','잔차 분포와 CF calibration'),('monthly_errors','월별 오차와 bias'),
    ('feature_importance','특징 중요도'),('residual_correlation_peak','11~15시 발전소 간 residual 상관'),
    ('metadata_error','설비용량·학습 이력과 오차'),('daily_errors','일별 오차와 SMP 가중 오차')]
T0 = time.monotonic()

def log(message):
    print(f'[{time.monotonic()-T0:7.1f}s] {message}', flush=True)

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def local_ts(values):
    return pd.DatetimeIndex(pd.to_datetime(values)).tz_convert('Asia/Seoul').as_unit('ns')

def forecast_columns(provider):
    train_names=pq.read_schema(DATA/f'train/forecast_{provider}.parquet').names
    test_names=pq.read_schema(DATA/f'test/forecast_{provider}.parquet').names
    cols=[c for c in train_names if c not in [provider+'_station','ts','round']]
    assert set(cols).issubset(test_names), 'Train-only forecast feature found'
    return cols

def build_features(base,sites,forecast_dir='train'):
    """Same feature builder can serve validation and future test timestamps."""
    ts=local_ts(base.ts)
    site_positions=sites.index.get_indexer(base.site_id)
    assert (site_positions>=0).all()
    fields={}
    for c in METADATA:
        values=sites.index.to_numpy()[site_positions] if c=='site_id' else sites[c].to_numpy()[site_positions]
        if c in CATEGORICAL:
            categories=list(sites.index if c=='site_id' else sites[c].unique())
            fields[c]=pd.Categorical(values,categories=categories)
        else: fields[c]=values.astype('float32')
    hour=ts.hour.to_numpy()
    day=ts.dayofyear.to_numpy()
    fields.update(hour=hour.astype('int8'),month=ts.month.to_numpy().astype('int8'),
        dayofyear=day.astype('int16'),hour_sin=np.sin(2*np.pi*hour/24).astype('float32'),
        hour_cos=np.cos(2*np.pi*hour/24).astype('float32'),
        dayofyear_sin=np.sin(2*np.pi*(day-1)/365.25).astype('float32'),
        dayofyear_cos=np.cos(2*np.pi*(day-1)/365.25).astype('float32'))
    quality=[]
    for provider in ['wa','wb']:
        station=provider+'_station'; cols=forecast_columns(provider)
        path=DATA/f'{forecast_dir}/forecast_{provider}.parquet'
        weather=pd.read_parquet(path,columns=[station,'round','ts']+cols)
        weather=weather.loc[weather['round']==17,[station,'ts']+cols].copy()
        weather['ts']=local_ts(weather.ts)
        assert not weather.duplicated([station,'ts']).any()
        index=pd.MultiIndex.from_frame(weather[[station,'ts']])
        lookup=pd.MultiIndex.from_arrays([np.asarray(fields[station]),ts],names=[station,'ts'])
        positions=index.get_indexer(lookup)
        available=positions>=0
        safe=np.maximum(positions,0)
        quality.append(dict(feature=provider+'_forecast_row',missing_rows=int((~available).sum()),total_rows=len(base)))
        for c in cols:
            arr=weather[c].to_numpy(dtype='float32')[safe]
            arr[~available]=np.nan
            arr[~np.isfinite(arr)]=np.nan
            fields[c]=arr
            quality.append(dict(feature=c,missing_rows=int(np.isnan(arr).sum()),total_rows=len(base)))
        del weather,index,lookup,positions,safe
        gc.collect()
    x=pd.DataFrame(fields)
    assert not set(x.columns)&{'energy','target_cf','actual_cf','smp','k','current_group'}
    assert list(x.columns)==METADATA+TIME_FEATURES+forecast_columns('wa')+forecast_columns('wb')
    return x,pd.DataFrame(quality)

def metrics_frame(pred,smp,sites):
    d=pred.merge(smp[['ts','smp']],on='ts',how='left',validate='many_to_one')
    assert d.smp.notna().all() and (d.smp>=0).all()
    d['hour']=local_ts(d.ts).hour
    d['peak']=d.hour.between(11,15)
    d['k']=np.where(d.peak,3.,1.)
    d['weight']=d.smp*d.k
    d['err_cf']=d.actual_cf-d.pred_cf
    d['abs_cf']=d.err_cf.abs()
    d['abs_kwh']=d.residual_kwh.abs()
    d['squared_kwh']=d.residual_kwh**2
    d['squared_cf']=d.err_cf**2
    d['weighted_error']=d.weight*d.abs_kwh
    return d

def summarize(d,keys):
    g=d.groupby(keys,observed=True)
    s=g.agg(observed_hours=('actual_cf','size'),cf_mae=('abs_cf','mean'),
        energy_mae_kwh=('abs_kwh','mean'),weighted_error=('weighted_error','sum'),
        weight_sum=('weight','sum'),bias_kwh=('residual_kwh','mean'),
        cf_bias=('err_cf','mean'),mse_kwh=('squared_kwh','mean'),cf_mse=('squared_cf','mean'))
    s['rmse_kwh']=np.sqrt(s.pop('mse_kwh')); s['cf_rmse']=np.sqrt(s.pop('cf_mse'))
    s['weighted_mae_kwh']=s.weighted_error/s.weight_sum.replace(0,np.nan)
    peak=d[d.peak].groupby(keys,observed=True).agg(peak_observed_hours=('actual_cf','size'),
        peak_cf_mae=('abs_cf','mean'),peak_energy_mae_kwh=('abs_kwh','mean'))
    return s.join(peak)

def render_report(out,manifest,overall,cohort,per_site,importance,plots,residual_summary=None):
    m=overall.iloc[0]
    texts=[
        f"모델: CatBoost global regressor 1개. round=17 하루 전 기상예보, 시간 특징, 발전소 메타데이터만 사용했습니다. observed weather, 발전량 lag, SMP, current_group은 특징에 포함하지 않았습니다. 전체 {len(manifest['features'])}개 특징입니다.",
        f"학습: {manifest['train_start']} ~ {manifest['train_end']} ({manifest['train_rows']:,}행, {manifest['train_sites']}개소). 검증: {manifest['validation_start']} ~ {manifest['validation_end']} ({manifest['validation_rows']:,}행, {manifest['validation_sites']}개소). 모든 학습 시각이 모든 검증 시각보다 이릅니다. 고정 {manifest['parameters']['iterations']}회 학습하며 holdout으로 조기종료·하이퍼파라미터 선택을 하지 않았습니다. CatBoost has_time=True로 시간순 입력을 보존합니다.",
        "generation에 실제로 존재하고 유한한 관측 행만 target_cf=energy/capacity_kw로 사용합니다. 첫 관측 이전 공백과 PV051의 내부 7시간은 제외되며 imputation은 없습니다. 음수·CF>1 target 관측도 원본대로 유지했습니다. 예보 결측은 train에서 계산한 변수별 중앙값으로 모델 입력에서만 대체하며 test/validation 통계를 사용하지 않습니다. 범주형 site_id/wa_station/wb_station은 정적 sites 메타로 정의합니다.",
        f"CF MAE={m.cf_mae:.5f}, energy MAE={m.energy_mae_kwh:.3f} kWh, 11~15시 CF MAE={m.peak_cf_mae:.5f}, 11~15시 energy MAE={m.peak_energy_mae_kwh:.3f} kWh입니다. weighted error=Σ SMP×k×|actual_kwh−pred_kwh|={m.weighted_error:,.3f}; k는 KST 우측닫힘 시각 11~15에 3, 나머지는 1입니다. weighted MAE는 weighted error / Σ SMP×k = {m.weighted_mae_kwh:.3f} kWh입니다. weighted error는 금액×시간대 가중 스케일의 미정규화 합계이며 정산식의 0.10은 곱하지 않았습니다.",
        f"학습 관측이 없는 신규 발전소 {manifest['cold_start_sites']}개를 포함합니다. 검증 초반에 운영 전인 발전소의 공백은 평가 대상에서 제외합니다. 따라서 검증 전체 253개소가 모든 시각에 공통으로 존재하지는 않습니다. 발전소별 train/validation 관측 수와 cold_start를 site_metrics.csv에 기록했습니다. 단일 시간 holdout 결과이므로 다른 연도에 대한 일반화나 모든 신규 발전소의 안정성을 보장하지 않습니다.",
        f"residual_kwh=actual_kwh−pred_kwh입니다. 양수 bias는 과소예측, 음수 bias는 과대예측입니다. 전체 residual bias는 {m.bias_kwh:.3f} kWh로 약한 과대예측이며 RMSE는 {m.rmse_kwh:.3f} kWh입니다. CF bias는 {m.cf_bias:.5f}로 양수입니다. CF 평균과 kWh 평균의 부호 차이는 발전소별 설비용량 가중 차이 때문입니다. RMSE가 MAE보다 커 큰 오차의 기여를 함께 살펴야 합니다. 추론의 CF는 음수 예측만 0으로 제한하고 상한 clipping이나 야간 강제 0은 사용하지 않았습니다. validation_predictions.parquet에는 요청한 7개 열만 있고 모델은 해당 구간의 발전량을 학습에 사용하지 않았습니다.",
        "잔차 상관은 발전소쌍에서 실제로 동시에 관측된 검증 시각만 사용합니다. NaN을 0으로 대체하지 않습니다. all-hour와 11~15시 상관 및 공통 관측 수를 함께 저장하며 상관 최소 관측 수는 168입니다. 밤의 거의 0인 잔차와 설비 편입 시점 차이가 상관에 영향을 줄 수 있습니다. 아직 G1/G2 구성·배정·최적화·제출 파일을 만들지 않았습니다. 향후 이 holdout을 그룹 최적화에 사용하면 그 이후에는 같은 구간을 최종 성능 검증으로 재사용하지 않아야 합니다.",
        "모든 입력 파일의 SHA-256을 실행 전후 확인했습니다. model.cbm은 holdout 이전 자료로만 학습한 검증용 모델이며, 전체 학습 구간으로 재학습한 배포용 모델은 아닙니다. 추론에는 feature_manifest.json의 범주 목록·train 중앙값·특징 순서와 동일한 feature builder를 사용해야 합니다."
    ]
    if residual_summary is not None:
        r=residual_summary
        texts.insert(7,f"11~15시 residual의 발전소쌍 평균 상관은 {r['peak_residual_correlation']['mean']:.3f}입니다. A 또는 B 기상지점을 공유하는 {r['peak_same_station_correlation']['pairs']:,}쌍은 평균 {r['peak_same_station_correlation']['mean']:.3f}, 공유하지 않는 {r['peak_different_station_correlation']['pairs']:,}쌍은 평균 {r['peak_different_station_correlation']['mean']:.3f}입니다. 지점을 공유하는 발전소의 오차가 함께 움직이는 특성이 관찰됩니다. 전체 시간별 kWh residual의 1~99% 분위수는 {r['residual_kwh_quantiles']['0.01']:.3f} ~ {r['residual_kwh_quantiles']['0.99']:.3f} kWh입니다. 대용량 발전소와 극단 오차의 영향은 발전소별 지표와 함께 해석해야 합니다.")
    tables=[('전체 지표',overall),('기존/신규 발전소 비교',cohort.reset_index()),
        ('energy MAE 상위 15개소',per_site.sort_values('energy_mae_kwh',ascending=False).head(15).reset_index()),
        ('CF MAE 상위 15개소',per_site.sort_values('cf_mae',ascending=False).head(15).reset_index()),
        ('특징 중요도 상위 20개',importance.head(20))]
    body='<h1>하루 전 기상예보 기반 CF baseline</h1>'+''.join('<p>'+html.escape(t)+'</p>' for t in texts)
    md='# 하루 전 기상예보 기반 CF baseline\n\n'+'\n\n'.join(texts)+'\n\n'
    for title,table in tables:
        body+='<h2>'+title+'</h2><div class="table">'+table.to_html(index=False,float_format=lambda x:f'{x:.5f}',na_rep='—')+'</div>'
        formatted=table.copy()
        for c in formatted:
            if pd.api.types.is_float_dtype(formatted[c]):
                formatted[c]=formatted[c].map(lambda v:'—' if pd.isna(v) else f'{v:.5f}')
        markdown='| '+' | '.join(map(str,formatted.columns))+' |\n| '+' | '.join(['---']*len(formatted.columns))+' |\n'
        markdown+='\n'.join('| '+' | '.join(map(str,row))+' |' for row in formatted.itertuples(index=False,name=None))
        md+='## '+title+'\n\n'+markdown+'\n\n'
    for name,title in plots:
        body+='<h2>'+title+'</h2><img src="'+name+'.png" alt="'+title+'">'
        md+='## '+title+'\n\n!['+title+']('+name+'.png)\n\n'
    body+='<h2>결과 파일</h2><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(out.iterdir()) if p.suffix in ['.csv','.json','.parquet','.cbm'])+'</ul>'
    (out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>CF baseline</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.7;color:#182638}img{max-width:100%}.table{overflow:auto}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #dde4ec;padding:6px;white-space:nowrap}h2{margin-top:40px}</style>'+body+'</html>',encoding='utf-8')
    (out/'report.md').write_text(md,encoding='utf-8')

def plot_results(out,d,per_site,importance,correlation):
    plots=[]
    def save(name,title):
        plt.tight_layout(); plt.savefig(out/(name+'.png'),dpi=140,bbox_inches='tight'); plt.close()
        plots.append((name,title))
    plt.rcParams.update({'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    h=d.groupby('hour').agg(actual=('actual_cf','mean'),pred=('pred_cf','mean'),mae=('abs_cf','mean'),bias=('err_cf','mean'))
    h[['actual','pred']].plot(ax=axes[0]); axes[0].set(xlabel='KST interval-end hour',ylabel='Mean CF')
    h[['mae','bias']].plot(ax=axes[1]); axes[1].set(xlabel='KST interval-end hour',ylabel='CF error')
    save('hourly_profile','시간별 CF 예측 및 오차')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    per_site.sort_values('cf_mae',ascending=False).head(20).cf_mae.plot.bar(ax=axes[0])
    per_site.sort_values('energy_mae_kwh',ascending=False).head(20).energy_mae_kwh.plot.bar(ax=axes[1])
    axes[0].set_ylabel('CF MAE'); axes[1].set_ylabel('Energy MAE (kWh)')
    save('site_errors','발전소별 MAE 상위 20개소')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    lim=np.quantile(d.residual_kwh,[.005,.995])
    axes[0].hist(d.residual_kwh,bins=100,range=lim); axes[0].set(xlabel='Actual − prediction (kWh), p0.5–p99.5 display',ylabel='Count')
    bins=np.linspace(max(0,float(d.pred_cf.min())),max(1,float(d.pred_cf.max())),21)
    calibration=d.groupby(pd.cut(d.pred_cf,bins=bins,include_lowest=True),observed=True).agg(pred=('pred_cf','mean'),actual=('actual_cf','mean'))
    axes[1].plot(calibration.pred,calibration.actual,'o-'); axes[1].plot([0,1],[0,1],'k--'); axes[1].set(xlabel='Mean predicted CF',ylabel='Mean actual CF')
    save('residual_calibration','잔차 분포와 CF calibration')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    d.groupby('month_label')[['abs_cf','err_cf']].mean().rename(columns={'abs_cf':'CF MAE','err_cf':'CF bias'}).plot(ax=axes[0],rot=45)
    d.groupby('month_label')[['abs_kwh','residual_kwh']].mean().rename(columns={'abs_kwh':'kWh MAE','residual_kwh':'kWh bias'}).plot(ax=axes[1],rot=45)
    save('monthly_errors','월별 오차와 bias')
    plt.figure(figsize=(10,7)); importance.head(20).iloc[::-1].plot.barh(x='feature',y='importance',ax=plt.gca(),legend=False)
    plt.xlabel('CatBoost PredictionValuesChange'); save('feature_importance','특징 중요도')
    plt.figure(figsize=(10,8)); plt.imshow(correlation,aspect='auto',vmin=-1,vmax=1,cmap='RdBu_r'); plt.colorbar(label='Pairwise residual correlation')
    plt.xlabel('Site order PV001–PV253'); plt.ylabel('Site order PV001–PV253'); save('residual_correlation_peak','11~15시 발전소 간 residual 상관')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for cold,label in [(False,'Seen in training'),(True,'Cold start')]:
        s=per_site[per_site.cold_start==cold]
        axes[0].scatter(s.capacity_kw,s.energy_mae_kwh,s=16,alpha=.7,label=label)
        axes[1].scatter(s.train_observed_hours,s.cf_mae,s=16,alpha=.7,label=label)
    axes[0].set(xlabel='Capacity (kW)',ylabel='Energy MAE (kWh)',xscale='log'); axes[0].legend()
    axes[1].set(xlabel='Training observed hours',ylabel='CF MAE'); axes[1].legend()
    save('metadata_error','설비용량·학습 이력과 오차')
    # Daily group totals are diagnostic only; no G1/G2 assignment is performed.
    daily=d.groupby('delivery_day').agg(abs_error=('abs_kwh','sum'),weighted_error=('weighted_error','sum'),observed_hours=('actual_cf','size'))
    daily['mae']=daily.abs_error/daily.observed_hours
    fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True)
    axes[0].plot(daily.index,daily.mae); axes[0].set_ylabel('Per-site hourly MAE (kWh)')
    axes[1].plot(daily.index,daily.weighted_error); axes[1].set_ylabel('Daily Σ SMP × k × |error|')
    save('daily_errors','일별 오차와 SMP 가중 오차')
    return plots

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation-start',default='2024-06-01 01:00')
    parser.add_argument('--iterations',type=int,default=400)
    parser.add_argument('--threads',type=int,default=6)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    assert args.iterations>0 and args.threads>0
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    source=[DATA/'sites.csv',DATA/'train/generation.parquet',DATA/'train/smp.csv']+[
        DATA/f'{folder}/forecast_{p}.parquet' for folder in ['train','test'] for p in ['wa','wb']]
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in source}
    sites=pd.read_csv(DATA/'sites.csv').set_index('site_id')
    assert sites.index.is_unique and (sites.capacity_kw>0).all()
    base=pd.read_parquet(DATA/'train/generation.parquet').sort_values(['ts','site_id']).reset_index(drop=True)
    base['ts']=local_ts(base.ts)
    assert not base.duplicated(['site_id','ts']).any()
    assert base.site_id.isin(sites.index).all()
    finite=np.isfinite(base.energy)
    excluded_nonfinite=int((~finite).sum())
    base=base.loc[finite].reset_index(drop=True)
    pv_gap=(base.site_id=='PV051')&base.ts.between(pd.Timestamp('2021-06-15 18:00',tz='Asia/Seoul'),pd.Timestamp('2021-06-16 00:00',tz='Asia/Seoul'))
    assert not pv_gap.any(), 'Unexpected observations inside documented PV051 gap'
    cut=pd.Timestamp(args.validation_start,tz='Asia/Seoul')
    assert (cut-pd.Timedelta(hours=1)).hour==0, 'Split must preserve right-closed delivery days'
    train=base.ts<cut; val=~train
    assert train.any() and val.any() and base.loc[train,'ts'].max()<base.loc[val,'ts'].min()
    capacity=base.site_id.map(sites.capacity_kw).to_numpy()
    target=base.energy.to_numpy()/capacity
    log(f'Observed targets only; train {train.sum():,}, holdout {val.sum():,}; preparing round-17 forecasts')
    x,quality=build_features(base,sites)
    quality.to_csv(out/'feature_missingness.csv',index=False)
    numerical=[c for c in x if c not in CATEGORICAL]
    medians=x.loc[train,numerical].median()
    if medians.isna().any(): raise ValueError('Feature completely missing in training')
    x[numerical]=x[numerical].fillna(medians)
    assert x[numerical].notna().all().all()
    audit=pd.DataFrame({'feature':x.columns,'source':['sites.csv' if c in METADATA else ('timestamp' if c in TIME_FEATURES else 'forecast_'+c[:2]+' round=17') for c in x],
        'available_at_test':True,'categorical':[c in CATEGORICAL for c in x]})
    audit.to_csv(out/'feature_audit.csv',index=False)
    params=dict(iterations=args.iterations,depth=7,learning_rate=.06,loss_function='RMSE',
        eval_metric='MAE',random_seed=20261008,thread_count=args.threads,has_time=True,
        one_hot_max_size=300,border_count=128,l2_leaf_reg=5.,allow_writing_files=False,verbose=50)
    log(f'Fitting one global CatBoost; {len(x.columns)} features; fixed {args.iterations} iterations')
    train_pool=Pool(x.loc[train],label=target[train],cat_features=CATEGORICAL)
    model=CatBoostRegressor(**params)
    model.fit(train_pool)
    model.save_model(str(out/'model.cbm'))
    importance=pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(out/'feature_importance.csv',index=False)
    del train_pool; gc.collect()
    raw=model.predict(Pool(x.loc[val],cat_features=CATEGORICAL))
    cf_pred=np.maximum(raw,0.)
    pred=base.loc[val,['site_id','ts']].copy()
    pred['actual_cf']=target[val]; pred['pred_cf']=cf_pred
    pred['actual_kwh']=base.loc[val,'energy'].to_numpy()
    pred['pred_kwh']=cf_pred*capacity[val]
    pred['residual_kwh']=pred.actual_kwh-pred.pred_kwh
    columns=['site_id','ts','actual_cf','pred_cf','actual_kwh','pred_kwh','residual_kwh']
    assert pred.columns.tolist()==columns
    assert np.isfinite(pred[columns[2:]].to_numpy()).all()
    pred.to_parquet(out/'validation_predictions.parquet',index=False,compression='zstd')
    del x; gc.collect()
    train_counts=base.loc[train].groupby('site_id').size().reindex(sites.index,fill_value=0)
    smp=pd.read_csv(DATA/'train/smp.csv'); smp['ts']=local_ts(smp.ts)
    assert smp.ts.is_unique
    d=metrics_frame(pred,smp,sites)
    d['cohort']=np.where(d.site_id.map(train_counts)>0,'seen','cold_start')
    # The timestamp labels the END of the hour; midnight belongs to prior delivery day.
    d['delivery_day']=(local_ts(d.ts)-pd.Timedelta(hours=1)).normalize()
    uniq=pd.DatetimeIndex(d.delivery_day.unique())
    d['month_label']=uniq.strftime('%Y-%m').to_numpy()[uniq.get_indexer(d.delivery_day)]
    d['all']='all'
    overall=summarize(d,['all']).reset_index()
    per_site=summarize(d,['site_id']).reindex(sites.index).join(sites[METADATA[3:]])
    per_site['train_observed_hours']=train_counts
    per_site['cold_start']=train_counts==0
    per_site['first_observed']=base.groupby('site_id').ts.min()
    per_site['last_observed']=base.groupby('site_id').ts.max()
    per_site['wa_station']=sites.wa_station; per_site['wb_station']=sites.wb_station
    cohort=summarize(d,['cohort'])
    overall.to_csv(out/'validation_metrics.csv',index=False)
    per_site.to_csv(out/'site_metrics.csv')
    cohort.to_csv(out/'cohort_metrics.csv')
    summarize(d,['month_label']).to_csv(out/'monthly_metrics.csv')
    summarize(d,['hour']).to_csv(out/'hourly_metrics.csv')
    log('Holdout predictions and metrics saved; analyzing residual correlations')
    wide=pred.pivot(index='ts',columns='site_id',values='residual_kwh').reindex(columns=sites.index)
    corr_peak=None
    for scope,frame in [('all',wide),('peak',wide.loc[wide.index.hour.isin(range(11,16))])]:
        corr=frame.corr(min_periods=168)
        corr.to_csv(out/f'residual_correlation_{scope}.csv')
        present=frame.notna().to_numpy(dtype='float64')
        pd.DataFrame(present.T@present,index=sites.index,columns=sites.index).astype('int32').to_csv(out/f'residual_overlap_{scope}.csv')
        if scope=='peak': corr_peak=corr
    manifest=dict(model='CatBoostRegressor global',catboost_version=catboost.__version__,round=17,
        features=audit.feature.tolist(),categorical_features=CATEGORICAL,
        category_levels={'site_id':sites.index.tolist(),'wa_station':sites.wa_station.unique().tolist(),'wb_station':sites.wb_station.unique().tolist()},
        numeric_train_medians={c:float(v) for c,v in medians.items()},parameters=params,
        train_start=str(base.loc[train,'ts'].min()),train_end=str(base.loc[train,'ts'].max()),
        validation_start=str(base.loc[val,'ts'].min()),validation_end=str(base.loc[val,'ts'].max()),
        train_rows=int(train.sum()),validation_rows=int(val.sum()),train_sites=int(base.loc[train,'site_id'].nunique()),
        validation_sites=int(pred.site_id.nunique()),cold_start_sites=int((train_counts==0).sum()),
        excluded_nonfinite_targets=excluded_nonfinite,negative_target_observations=int((target<0).sum()),
        cf_above_1_observations=int((target>1).sum()),negative_raw_predictions=int((raw<0).sum()),
        residual_sign='actual_kwh - pred_kwh',prediction_postprocess='max(raw_cf, 0); no upper clipping',
        imputation='no generation imputation; forecast missing features use TRAIN-only medians',
        weighted_error_formula='sum(smp * (3 if 11<=KST hour<=15 else 1) * abs(residual_kwh))',
        input_sha256=hashes)
    assert all(sha256(p)==hashes[str(p.relative_to(ROOT))] for p in source), 'Source file changed'
    (out/'feature_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    plots=plot_results(out,d,per_site,importance,corr_peak)
    render_report(out,manifest,overall,cohort,per_site,importance,plots)
    log(overall.to_string(index=False)); log(cohort.to_string())
    log(f'Report: {out / "report.html"}')

if __name__=='__main__': main()
