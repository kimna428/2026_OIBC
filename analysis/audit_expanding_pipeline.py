"""Independently audit saved expanding predictions and enrich the final report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from expanding_pipeline import OUT, DATA, issuance, operating_day, sha256, markdown, report

def main(out=OUT):
    out=Path(out)
    config=json.loads((out/'config.json').read_text())
    assert all(sha256(DATA/p)==v for p,v in config['input_hashes'].items())
    g=pd.read_parquet(DATA/'train/generation.parquet').set_index(['site_id','ts']).energy
    sites=pd.read_csv(DATA/'sites.csv').set_index('site_id')
    folds=pd.read_csv(out/'split_audit.csv')
    fits=json.loads((out/'fits.json').read_text())
    checks=[]; periods=[]
    predictions=list(out.glob('fold*_predictions.parquet'))
    assert len(predictions)==42
    reference={}
    import lightgbm as lgb
    prepared=pd.read_parquet(out/'datasets/V3.parquet')
    prepared['site_id_code']=pd.Categorical(prepared.site_id,categories=list(sites.index)).codes.astype('int16')
    prepared['capacity_kw']=prepared.capacity_kw.astype('float32')
    prepared_index=pd.MultiIndex.from_frame(prepared[['site_id','ts']])
    for path in sorted(predictions):
        fi=int(path.name.split('_')[0][4:]); d=pd.read_parquet(path)
        assert not d.duplicated(['site_id','ts']).any()
        actual=g.reindex(pd.MultiIndex.from_frame(d[['site_id','ts']])).to_numpy()
        np.testing.assert_array_equal(actual,d.energy)
        np.testing.assert_allclose(d.pred_cf*d.capacity_kw,d.pred_kwh)
        assert np.isfinite(d.pred_kwh).all() and (d.pred_kwh>=0).all()
        keys=pd.MultiIndex.from_frame(d[['site_id','ts']])
        if fi in reference: assert keys.equals(reference[fi])
        else: reference[fi]=keys
        cutoff=pd.Timestamp(folds.loc[folds.fold==fi,'training_cutoff'].iloc[0])
        assert (cutoff<=issuance(d.ts)).all()
        replay_pass=None
        if '_A_climatology_' not in path.name:
            sample=d.iloc[np.linspace(0,len(d)-1,min(64,len(d)),dtype=int)]
            pos=prepared_index.get_indexer(pd.MultiIndex.from_frame(sample[['site_id','ts']]))
            assert (pos>=0).all()
            model=lgb.Booster(model_file=str(out/(path.name.removesuffix('_predictions.parquet')+'.txt')))
            features=model.feature_name()
            xx=prepared.iloc[pos][features].copy()
            xx['site_id']=prepared.site_id_code.iloc[pos].to_numpy()
            pred=np.maximum(model.predict(xx,num_threads=4),0)
            if '_V0_B_' in path.name: pred=pred/sample.capacity_kw.to_numpy()
            np.testing.assert_allclose(pred,sample.pred_cf,rtol=1e-10,atol=1e-12)
            replay_pass=True
        checks.append({'file':path.name,'rows':len(d),'original_target_match':True,'finite_nonnegative_prediction':True,'cutoff_pass':True,'model_sample_replay':replay_pass})
    raw=g.reset_index()
    for row in folds.itertuples():
        cutoff=pd.Timestamp(row.training_cutoff)
        train=raw[raw.ts<=cutoff].groupby('site_id').ts.agg(['min','max','size']).reindex(sites.index)
        train['fold']=row.fold; train['cohort']=np.where(train['size'].fillna(0)>=2160,'established_90d','new_under90d')
        periods.append(train.reset_index())
    pd.concat(periods).to_csv(out/'fold_site_training_periods.csv',index=False)
    # Persist exact fitting feature allowlists, targets, schemas and their availability contracts.
    manifest={}
    for version in ['V0','V1','V2','V3']:
        path=out/'datasets'/f'{version}.parquet'
        schema=pq.read_schema(path)
        assert {'site_id','ts','target','train_eligible'}.issubset(schema.names)
        experiment=version+'_B'
        features=next(f['features'] for f in fits if f['experiment']==experiment)
        assert not set(features)&{'energy','target','cf','train_eligible','negative','cf_gt_1','smp','k'}
        manifest[version]={'file':str(path.relative_to(out)),'schema':{f.name:str(f.type) for f in schema},'features':features,'target_units':'kWh' if version=='V0' else 'CF','note':'V0/V1/V2 use deterministic feature builder. V3 audit site_id remains string: encode to sites.csv row-order int16 codes and cast capacity_kw float32 before model prediction; other category codes already materialized.'}
    manifest['V3_C']={'features':next(f['features'] for f in fits if f['experiment']=='V3_C'),'history_availability':'lag48 source <= issuance; rolling168 ends at issuance; no observed-weather input'}
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Extract the authoritative variable-unit tables from the provided README.
    readme=(DATA/'README.md').read_text()
    units={}
    for line in readme.splitlines():
        if line.startswith('| `wa_') or line.startswith('| `wb_'):
            cells=[c.strip() for c in line.strip('|').split('|')]
            if len(cells)>=3: units[cells[0].strip('`')]=cells[2]
    quality=pd.read_csv(out/'weather_quality.csv'); quality['unit']=quality.variable.map(units)
    assert quality.unit.notna().all()
    quality.to_csv(out/'weather_quality.csv',index=False)
    m=pd.read_csv(out/'metrics.csv'); overall=m[m.dimension=='overall'].pivot(index='fold',columns='experiment',values='cf_mae')
    pairs=[('V0_B','V1_B'),('V1_B','V2_B'),('V2_B','V2_interp_B'),('V2_B','V3_B'),('V3_B','V3_C')]
    ablation=pd.DataFrame([dict(reference=a,experiment=b,mean_fold_cf_mae_delta=float((overall[b]-overall[a]).mean()),improved_folds=int((overall[b]<overall[a]).sum()),equal_folds=int(np.isclose(overall[b],overall[a],rtol=0,atol=1e-12).sum())) for a,b in pairs])
    ablation.to_csv(out/'ablation.csv',index=False)
    score=pd.read_csv(out/'official_formula_scores.csv')
    cohort=m[m.dimension=='cohort'].groupby(['experiment','label']).apply(lambda d:np.average(d.cf_mae,weights=d.rows),include_groups=False).unstack()
    spring=pd.read_csv(out/'spring_comparison.csv')
    report(out,m,score.set_index(['fold','experiment']),config)
    text=(out/'report.md').read_text().split('\n## 직접 비교한 Ablation')[0]
    text+='\n## 직접 비교한 Ablation\n\n'+markdown(ablation)+'\n\n음수 차이는 개선입니다. 20만 표본은 버전 간 고정했으며 이상치 제외는 그 표본에만 적용했습니다. V2/V3 전체 데이터 파일에서는 모든 이상치가 train_eligible=False입니다.\n'
    text+='\n## 공식 산식 로컬 점수\n\n'+markdown(score)+'\n\n전체 253개소 관측이 있는 시간에만 계산합니다. 초기 fold의 낮은 관측 범위에서는 집합 점수를 계산할 수 없고, 나머지 지표는 원본 관측만 평가합니다.\n'
    official=score.pivot(index='fold',columns='experiment',values='score_krw')
    text+='\nC와 B의 정산 점수 차이(C−B)는 '
    text+=', '.join(f'fold {fi}: {row.V3_C-row.V3_B:+,.0f}원' for fi,row in official.iterrows())
    text+='입니다. 과거 실적 피처가 CF MAE를 개선해도 정산 점수가 항상 개선되는 것은 아닙니다.\n'
    text+='\n## 신규/기존 CF MAE (검증 관측 수 가중)\n\n'+markdown(cohort)+'\n'
    text+='\n## 2025 봄 변화 해석\n\n'
    for r in spring.itertuples():
        text+=f'- {r.other_year} 대비 {r.paired_sites}개 공통 발전소의 2025 평균 CF 차이 {r.mean_cf_delta:+.5f}, 발전소 bootstrap 95% CI [{r.ci_low:+.5f}, {r.ci_high:+.5f}], 보정 p={r.bonferroni_p:.3g}.\n'
    text+='\n2021년 비교는 공통 발전소 1개뿐이어서 모집단에 대한 통계적 결론을 내릴 수 없습니다. bootstrap 구간이 퇴화한 것은 충분한 확실성을 의미하지 않습니다.\n'
    text+='\n실측 기상 A 키 중복은 weather_cadence.csv에 기록되어 있습니다. 실측은 피처로 사용하지 않으므로 임의 중복 병합을 하지 않았습니다. CF 상관은 야간 공통 패턴의 영향이 커 주간(07~18시) 결과도 별도로 저장했습니다. 태양고도는 비식별 좌표와 근사식에 따른 값으로 실제 설비의 경사·방위·차폐를 반영하지 않습니다.\n'
    # Additional descriptive plots use only saved provided-data summaries.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.read_csv(out/'cf_month_hour.csv')
    grid=table.pivot(index='month',columns='hour',values='mean')
    fig,ax=plt.subplots(figsize=(10,4)); image=ax.imshow(grid,aspect='auto'); ax.set_yticks(range(12),range(1,13)); ax.set_xlabel('KST interval-end hour'); ax.set_ylabel('Operating month'); fig.colorbar(image,ax=ax,label='Mean CF'); fig.tight_layout(); fig.savefig(out/'cf_month_hour.png'); plt.close(fig)
    yearly=pd.read_csv(out/'cf_site_year.csv')
    fig,ax=plt.subplots(figsize=(9,4))
    for _,plant in yearly.groupby('site_id'): ax.plot(plant.year,plant['mean'],alpha=.12,color='steelblue')
    ax.set_ylabel('Observed mean CF (coverage varies)'); ax.set_xlabel('Operating year'); fig.tight_layout(); fig.savefig(out/'cf_site_year.png'); plt.close(fig)
    corr=pd.read_csv(out/'cf_daylight_correlation.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(7,6)); image=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); fig.colorbar(image,ax=ax,label='Daylight CF Pearson correlation'); fig.tight_layout(); fig.savefig(out/'cf_daylight_correlation.png'); plt.close(fig)
    # Show hourly validation errors directly in the report, weighted by observed rows.
    h=m[m.dimension=='hour'].copy()
    h['label']=h.label.astype(int)
    hourly=h.groupby(['experiment','label']).apply(lambda z:np.average(z.cf_mae,weights=z.rows),include_groups=False).unstack('experiment').sort_index()
    hourly.to_csv(out/'hourly_performance.csv')
    hourly[['A_climatology','V3_B','V3_C']].plot(figsize=(10,4),marker='o')
    plt.xlabel('KST interval-end hour'); plt.ylabel('CF MAE'); plt.tight_layout(); plt.savefig(out/'hourly_performance.png'); plt.close()
    ledger=pd.read_parquet(out/'settlement_ledger.parquet')
    ledger['abs_error_kwh']=(ledger.pred_kwh-ledger.energy).abs()
    grouped=ledger.groupby(['fold','experiment','group']).agg(scored_hours=('ts','size'),mae_kwh=('abs_error_kwh','mean'),score_krw=('score_krw','sum'),penalty_krw=('penalty_krw','sum')).reset_index()
    grouped.to_csv(out/'official_group_metrics.csv',index=False)
    text+='\n## Fold 기간과 입력 자료\n\n'+markdown(folds[['fold','validation_start','validation_end','training_cutoff','train_rows','validation_rows']])
    text+='\n\n'+markdown(pd.read_csv(out/'input_inventory.csv'))
    text+='\n\n전체 컬럼명과 dtype: [schemas.json](schemas.json). 기상 변수별 단위·결측률·가용 시점: [weather_quality.csv](weather_quality.csv). 실제 간격과 중복 키: [weather_cadence.csv](weather_cadence.csv). 발전소별 fold 학습 기간: [fold_site_training_periods.csv](fold_site_training_periods.csv).\n'
    text+='\n## 시간대별 검증 성능\n\n'+markdown(hourly[['A_climatology','V3_B','V3_C']])
    text+='\n\n![hourly validation errors](hourly_performance.png)\n'
    text+='\n집합별 MAE·정산 점수는 [official_group_metrics.csv](official_group_metrics.csv), 월별·발전소별·fold별 상세 오차는 [metrics.csv](metrics.csv)에 저장했습니다. 집합 지표는 원본 완전 관측 시간만 사용합니다.\n'
    text+='\n![season/hour](cf_month_hour.png)\n\n![yearly plants](cf_site_year.png)\n\n![correlation](cf_daylight_correlation.png)\n'
    text+='\n원본 유한 발전 관측 4,392,566행, 253개소, 내부 공백 7시간, 음수 40건, CF>1 9건입니다. 1시간 단독 공백이 없어 보간 추가 행은 0개이며, 이번 보간 ablation의 동률은 보간의 일반적인 효과를 검증한 결과가 아닙니다.\n'
    (out/'report.md').write_text(text)
    verification=json.loads((out/'verification.json').read_text())
    verification['independent_prediction_audit']=checks
    verification['dataset_schema_checks']=True
    verification['identical_validation_keys_all_versions']=True
    verification['generation_receipt_assumption']='Generation available at right-closed timestamp; receipt latency unspecified by provided data. C remains conditional on this assumption.'
    verification['source_hashes']={p.name:sha256(p) for p in Path(__file__).parent.glob('*expanding*.py')}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print('Audited',len(checks),'prediction files; all targets match raw input.')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--out',default=str(OUT))
    main(parser.parse_args().out)
