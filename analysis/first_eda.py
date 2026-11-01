"""Read-only first EDA. Run from any directory with .venv/bin/python analysis/first_eda.py."""
from pathlib import Path
import hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/oibc-first-eda-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'handover 2/01_participant_dataset'
OUT = ROOT / 'reports/data_quality'
TZ = 'Asia/Seoul'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def table(df):
    return df.to_markdown(index=False, floatfmt='.4f')
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[DATA/'sites.csv', DATA/'train/generation.parquet']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    sites=pd.read_csv(paths[0]).set_index('site_id').sort_index()
    g=pd.read_parquet(paths[1]); g.ts=pd.to_datetime(g.ts).dt.tz_convert(TZ)
    assert sites.index.is_unique and (sites.capacity_kw>0).all()
    assert g.site_id.isin(sites.index).all() and g.ts.notna().all()
    duplicates=int(g.duplicated(['site_id','ts']).sum())
    assert duplicates==0, 'Resolve duplicate keys explicitly before aggregation'
    assert ((g.ts.dt.minute==0)&(g.ts.dt.second==0)).all()
    grid=pd.date_range(g.ts.min(),g.ts.max(),freq='h')
    w=g.pivot(index='ts',columns='site_id',values='energy').reindex(index=grid,columns=sites.index)
    valid=w.notna() & np.isfinite(w)
    rows=[]
    for sid in sites.index:
        times=grid[valid[sid]]
        first,last=times.min(),times.max()
        expected=int((last-first)/pd.Timedelta(hours=1))+1 if len(times) else 0
        rows.append(dict(site_id=sid,first_observed=first,last_observed=last,total_rows=int((g.site_id==sid).sum()),finite_observed_hours=len(times),span_expected_hours=expected,span_missing_hours=expected-len(times),span_missing_rate=1-len(times)/expected if expected else np.nan,pre_first_hours=int((grid<first).sum()),post_last_hours=int((grid>last).sum())))
    cov=pd.DataFrame(rows).set_index('site_id').join(sites[['capacity_kw']])
    cov.to_csv(OUT/'site_coverage.csv')
    g['cf']=g.energy/g.site_id.map(sites.capacity_kw)
    stats=g.groupby('site_id').cf.agg(['count','mean','std','min','max'])
    stats=stats.join(g.groupby('site_id').cf.quantile([.01,.25,.5,.75,.95,.99,.999]).unstack().rename(columns={.01:'p01',.25:'p25',.5:'p50',.75:'p75',.95:'p95',.99:'p99',.999:'p999'}))
    for name,mask in {'negative':g.energy<0,'zero':g.energy==0,'cf_gt_1':g.cf>1,'cf_gt_1_2':g.cf>1.2,'nonfinite':~np.isfinite(g.energy)}.items():
        stats[name]=mask.groupby(g.site_id).sum()
    stats.to_csv(OUT/'site_capacity_factor.csv')
    anomalies=g[(g.energy<0)|(g.cf>1)|~np.isfinite(g.energy)].copy()
    anomalies.to_csv(OUT/'anomalous_observations.csv',index=False)
    # Mask finite observations only, preserve missing energy throughout.
    w=w.where(valid)
    active=pd.DataFrame({sid:(grid>=cov.loc[sid,'first_observed'])&(grid<=cov.loc[sid,'last_observed']) for sid in sites.index},index=grid)
    # Hour ending 00:00 belongs to the previous operating day/month.
    day=(grid-pd.Timedelta(nanoseconds=1)).normalize()
    month=(grid-pd.Timedelta(nanoseconds=1)).strftime('%Y-%m')
    monthly_n=valid.groupby(month).sum(); monthly_den=active.groupby(month).sum()
    monthly=monthly_n/monthly_den.replace(0,np.nan)
    monthly.T.to_csv(OUT/'monthly_active_observation_rate.csv')
    fig,ax=plt.subplots(figsize=(17,21))
    cmap=plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
    im=ax.imshow(monthly.T,aspect='auto',vmin=0,vmax=1,cmap=cmap)
    ax.set_yticks(range(len(sites)),sites.index,fontsize=4)
    ax.set_xticks(range(0,len(monthly),2),monthly.index[::2],rotation=60)
    ax.set_title('Monthly coverage within first-to-last observation span; grey = outside span')
    fig.colorbar(im,ax=ax,label='Observed / expected active hours'); fig.tight_layout(); fig.savefig(OUT/'monthly_coverage.png',dpi=160); plt.close(fig)
    daily=pd.DataFrame({'observed_energy_kwh':w.sum(axis=1,min_count=1).groupby(day).sum(min_count=1),'observed_site_hours':valid.sum(axis=1).groupby(day).sum(),'active_expected_site_hours':active.sum(axis=1).groupby(day).sum(),'available_grid_hours':pd.Series(1,index=grid).groupby(day).sum()})
    daily['active_coverage']=daily.observed_site_hours/daily.active_expected_site_hours.replace(0,np.nan)
    daily['full_fleet_expected_site_hours']=24*len(sites)
    daily['full_fleet_coverage']=daily.observed_site_hours/daily.full_fleet_expected_site_hours
    daily['possible_underaggregation_active']=daily.observed_site_hours<daily.active_expected_site_hours
    daily['complete_full_fleet_day']=daily.observed_site_hours==daily.full_fleet_expected_site_hours
    daily.to_csv(OUT/'daily_fleet_energy.csv')
    fig,axes=plt.subplots(2,1,figsize=(15,7),sharex=True)
    axes[0].plot(daily.index,daily.observed_energy_kwh/1000,lw=.7,label='Observed sum (no imputation)')
    bad=~daily.complete_full_fleet_day
    axes[0].scatter(daily.index[bad],daily.loc[bad,'observed_energy_kwh']/1000,s=3,c='red',label='Incomplete 253-site day: total may be understated')
    axes[0].set_ylabel('MWh'); axes[0].legend(fontsize=8)
    axes[1].plot(daily.index,daily.active_coverage,label='Within observed active spans')
    axes[1].plot(daily.index,daily.full_fleet_coverage,label='All 253 sites × 24 hours',alpha=.8)
    axes[1].set_ylabel('Coverage'); axes[1].legend(); fig.tight_layout(); fig.savefig(OUT/'daily_total_energy.png',dpi=150); plt.close(fig)
    # Per-site daily panels retained in a multipage PDF for readable inspection.
    from matplotlib.backends.backend_pdf import PdfPages
    ds=w.groupby(day).sum(min_count=1); dn=valid.groupby(day).sum(); de=active.groupby(day).sum()
    ds.to_csv(OUT/'daily_site_energy_kwh.csv'); (dn/de.replace(0,np.nan)).to_csv(OUT/'daily_site_active_coverage.csv')
    with PdfPages(OUT/'daily_by_site.pdf') as pdf:
        for start in range(0,len(sites),12):
            fig,axs=plt.subplots(4,3,figsize=(15,10),sharex=True)
            for ax,sid in zip(axs.flat,sites.index[start:start+12]):
                ax.plot(ds.index,ds[sid],lw=.5)
                incomplete=(dn[sid]<24)&ds[sid].notna()
                ax.scatter(ds.index[incomplete],ds.loc[incomplete,sid],s=2,c='red')
                ax.set_title(sid,fontsize=9); ax.tick_params(labelsize=6)
            fig.suptitle('Daily observed kWh; red = fewer than 24 finite hours; blanks = no observations')
            fig.autofmt_xdate(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    fig,ax=plt.subplots(figsize=(17,5))
    x=np.arange(len(stats)); ax.vlines(x,stats.p01,stats.p99,color='steelblue',lw=1,label='p01–p99'); ax.scatter(x,stats['max'],s=5,c='red',label='Maximum'); ax.scatter(x,stats['min'],s=4,c='black',label='Minimum'); ax.axhline(1,color='orange',ls='--'); ax.set_xticks(x[::5],stats.index[::5],rotation=90,fontsize=6); ax.set_ylabel('Hourly energy / capacity (1 hour)'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'capacity_factor_by_site.png',dpi=160); plt.close(fig)
    start=pd.Timestamp('2024-06-01',tz=TZ); end=pd.Timestamp('2025-06-01',tz=TZ)
    vg=pd.date_range(start+pd.Timedelta(hours=1),end,freq='h')
    v=valid.reindex(vg,fill_value=False)
    val=cov[['first_observed','last_observed']].copy()
    val['expected_hours']=len(vg); val['observed_hours']=v.sum(); val['observation_rate']=v.mean()
    val['observed_before_validation']=cov.first_observed<=start
    val['spans_entire_validation']=(cov.first_observed<=vg[0])&(cov.last_observed>=vg[-1])
    val['validation_active_expected_hours']=[int(((vg>=cov.loc[s,'first_observed'])&(vg<=cov.loc[s,'last_observed'])).sum()) for s in val.index]
    val['active_observation_rate']=val.observed_hours/val.validation_active_expected_hours.replace(0,np.nan)
    val['status']=np.select([val.observed_hours==0,~val.observed_before_validation,val.observation_rate>=.99,val.observation_rate>=.95],['no_targets','new_site_partial','existing_ge_99pct','existing_ge_95pct'],default='existing_lt_95pct')
    val.to_csv(OUT/'validation_site_coverage.csv')
    assert (cov.span_missing_hours>=0).all()
    assert int(cov.finite_observed_hours.sum())==int(valid.to_numpy().sum())
    assert int(daily.observed_site_hours.sum())==int(valid.to_numpy().sum())
    assert np.isclose(daily.observed_energy_kwh.sum(),g.energy.sum())
    assert all(sha(p)==hashes[str(p.relative_to(ROOT))] for p in paths)
    (OUT/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    summary={'rows':len(g),'sites':len(sites),'duplicates':duplicates,'nonfinite':int((~np.isfinite(g.energy)).sum()),'internal_missing_hours':int(cov.span_missing_hours.sum()),'span_missing_rate':float(cov.span_missing_hours.sum()/cov.span_expected_hours.sum()),'negative':int((g.energy<0).sum()),'cf_gt_1':int((g.cf>1).sum()),'cf_gt_1_2':int((g.cf>1.2).sum()),'max_cf':float(g.cf.max()),'validation_status':val.status.value_counts().to_dict()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Avoid optional tabulate dependency.
    def md(df):
        d=df.copy().fillna('NA'); return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
    text=f'''# 2026 OIBC 첫 데이터 품질 EDA

## 분석 기준

입력은 `sites.csv`, `train/generation.parquet`만 사용했다. KST, 우측닫힘 1시간 발전량(kWh)이다. 최초~최종 관측 사이의 매시간을 기대 관측으로 두고 유한 발전량이 있는 시각을 관측으로 센다. 결측률 = 1 − 유한 관측 시간 / 기대 시간. 행 수와 유효 관측 수를 별도로 저장했다. 중복 키 {duplicates}개, 비유한 발전량 {summary['nonfinite']}개이며 중복 발견 시 코드는 중단한다.

최초 관측 전은 **편입 전 추정·상태 불명**, 최종 관측 후는 **관측 종료 후·상태 불명**이다. 실제 편입/폐쇄일이 없으므로 이를 확정하지 않으며 운영 중 결측률의 분모에서 제외한다. 최초~최종 사이는 운영 구간의 대리 지표로 사용하며 통신 장애와 실제 운영 중단을 구분할 수 없다. 전체 비교 시간축은 두 입력에서 확인되는 {grid[0]} ~ {grid[-1]}이며 공식 학습 시작 이전/이후를 추측해 확장하지 않았다.

일별 집계는 `(ts − 1ns)`의 KST 날짜를 사용하여 00시를 전날에 배정한다. 첫날은 자료가 중간에 시작하는 경계일이다. 결측 발전량은 0으로 채우지 않았고, 전부 결측인 일 합계는 NaN이다. 일부만 있는 날의 합계는 **관측 부분합**이며 완전한 총량으로 해석하면 안 된다. 음수 관측도 그대로 보존하므로 부분합을 엄밀한 하한으로 단정하지 않는다.

## 핵심 발견 5개

1. {len(sites)}개 발전소, {len(g):,}행이다. 최초 관측은 {cov.first_observed.min()}, 가장 늦은 편입 추정은 {cov.first_observed.max()}로 발전소별 이력이 크게 다르다.
2. 최초~최종 관측 구간 내 결측은 총 {summary['internal_missing_hours']:,} site-hours, 가중 결측률 {summary['span_missing_rate']:.6%}이다. 편입 전 공백은 여기에 포함하지 않는다. 발전소별 결측률 중앙값 {cov.span_missing_rate.median():.2%}, 최대 {cov.span_missing_rate.max():.2%}이다.
3. 음수 발전량 {summary['negative']:,}건, 용량 대비 시간 발전량 비율(CF)>1 관측 {summary['cf_gt_1']:,}건, >1.2 관측 {summary['cf_gt_1_2']:,}건, 최대 CF {summary['max_cf']:.4f}이다. 계측 오류·용량 정의 차이 등을 추가 확인할 후보이며 자동 삭제/클리핑하지 않았다.
4. 일별 전체 {len(daily):,}일 중 253개소 × 24시간이 모두 있는 날은 {int(daily.complete_full_fleet_day.sum()):,}일이다. 관측 활성 구간에서도 누락이 있는 날은 {int(daily.possible_underaggregation_active.sum()):,}일이다. 편입 증가와 결측으로 인한 합계 변화가 발전량 추세에 섞인다.
5. 검증 8,760시간에서 사전 관측 이력이 있고 관측률 ≥99%인 발전소는 {int(((val.observed_before_validation)&(val.observation_rate>=.99)).sum())}개, ≥95%는 {int(((val.observed_before_validation)&(val.observation_rate>=.95)).sum())}개다. 검증 시작 이전 이력이 없는 신규 발전소는 {int((~val.observed_before_validation).sum())}개로 별도 평가해야 한다.

## 발전소별 관측과 결측

전 발전소 상세: [site_coverage.csv](data_quality/site_coverage.csv). `pre_first_hours`와 `post_last_hours`는 전체 비교 시간축 기준으로, 내부 결측과 구분한다.

결측률 상위 15개:

{md(cov.sort_values('span_missing_rate',ascending=False).head(15).reset_index()[['site_id','first_observed','last_observed','total_rows','span_expected_hours','span_missing_rate']])}

## 설비용량 대비 분포와 이상치

CF = energy(kWh) / (capacity_kw × 1h). [발전소별 평균·표준편차·분위수·최솟값·최댓값·이상치 수](data_quality/site_capacity_factor.csv), [이상치 원시 관측](data_quality/anomalous_observations.csv). 0은 실제 관측이며 결측과 다르다. 전 시간 분포에는 야간이 포함된다. >1은 검토 기준, >1.2는 추가 강도 기준이며 물리적으로 확정된 오류 판정이 아니다. 태양광의 주야·계절 혼합 분포에 단순 IQR 규칙을 적용하면 정상 주간 관측까지 이상치로 오인할 수 있다.

![CF 분포](data_quality/capacity_factor_by_site.png)

CF 최댓값 상위 10개:

{md(stats.sort_values('max',ascending=False).head(10).reset_index()[['site_id','mean','p50','p95','p99','min','max','negative','cf_gt_1','cf_gt_1_2']])}

## 월별 관측률과 일별 발전량

월별 분모는 해당 월과 최초~최종 관측 구간의 교집합이다. 회색은 그 구간 밖이며 0%와 다르다. 편입/종료 월은 부분월임에 유의한다. [월별 수치](data_quality/monthly_active_observation_rate.csv).

![월별 관측률](data_quality/monthly_coverage.png)

빨간 점은 253개소 전체의 24시간이 갖춰지지 않은 날이며, 편입 전 공백도 포함한다. 아래 두 관측률은 운영 구간 내부 누락과 고정된 전체 집합의 미관측을 구별한다. 활성 구간 관측률이 100%여도 전체 발전소 총량이 완전하다는 뜻은 아니다.

![일별 관측 부분합과 관측률](data_quality/daily_total_energy.png)

[일별 전체 부분합·결측 표시](data_quality/daily_fleet_energy.csv), [발전소별 일별 그래프 PDF](data_quality/daily_by_site.pdf), [발전소별 일별 부분합](data_quality/daily_site_energy_kwh.csv), [일별 활성 관측률](data_quality/daily_site_active_coverage.csv). PDF의 빨간 점은 해당 발전소 관측이 24시간보다 적은 날이다.

## 2024-06-01~2025-05-31 검증 가능성

운영일 기준 2024-06-01~2025-05-31은 우측닫힘 시각 **2024-06-01 01:00 ~ 2025-06-01 00:00 KST**, 8,760시간이다. 관측률은 이 전체 구간을 분모로 한다. 비교용 `active_observation_rate`는 검증 구간과 개별 관측 구간 교집합을 분모로 하므로 신규 발전소도 높을 수 있다.

유한 target이 있으면 그 시각에서 평가할 수 있지만, 전체 연간 검증의 대표성과 과거 학습 가능성은 별개다. ≥99%, ≥95%는 분석상 편의 기준이며 대회 공식 기준이 아니다. `observed_before_validation`은 검증 첫 구간 이전 관측이 있는지, `spans_entire_validation`은 양끝을 포괄하는지를 표시한다. 결측을 채워 평가하지 않는다.

상태별 발전소 수: {json.dumps(summary['validation_status'],ensure_ascii=False)}.

전체 253개소의 검증 가능성과 관측률:

{md(val.reset_index()[['site_id','observed_hours','observation_rate','active_observation_rate','observed_before_validation','status']])}

[검증 상세 CSV](data_quality/validation_site_coverage.csv).

## 모델 학습 시 주의할 점

- 관측된 유한 target만 기본 학습/평가에 사용한다. 편입 전과 내부 누락에 0을 넣으면 야간 및 발전량 수준을 왜곡한다.
- 시간순 분리를 사용하고 검증 관측으로 결측 복원·스케일링·이상치 임계값을 학습하지 않는다. 일별 집계와 경계도 KST 우측닫힘으로 통일한다.
- 신규 발전소는 cold-start 성능을 별도 보고한다. 기존 발전소의 연간 관측률이 높은 집합과 부분 관측 집합을 나누고 발전소별 유효 표본 수도 함께 제시한다.
- 음수/CF 초과를 우선 추적하고, 제거 또는 clipping은 학습 구간에서 근거를 정한 뒤 민감도 분석한다. 평가 target은 임의로 바꾸지 않는다.
- 관측된 시각만의 오차는 결측이 무작위가 아닐 때 편향될 수 있다. 월·시간대별 관측률과 성능을 함께 확인하고, 집합 발전량 평가는 구성원 관측 완전성을 확인한다. 부분 일 합계를 완전한 일 target으로 사용하지 않는다.

## 재현 및 검증

저장소 루트에서 `MPLCONFIGDIR=/tmp/oibc-first-eda-mpl .venv/bin/python analysis/first_eda.py` 실행. 의존성은 기존 `analysis/requirements.txt`의 pandas/numpy/matplotlib/pyarrow를 사용한다. 분석 코드는 두 입력만 읽는다. 실행 전후 SHA-256 일치, 키 유일성, 시간 정렬 단위, 유효 관측 수 보존, 일별 합계와 원관측 총합 일치를 assert로 검증했다. [입력 해시](data_quality/input_hashes.json), [기계 판독 요약](data_quality/summary.json). 원본을 수정하지 않으며 산출물만 덮어쓴다.
'''
    (ROOT/'reports/data_quality.md').write_text(text)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
