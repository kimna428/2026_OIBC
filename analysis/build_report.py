"""Build Korean HTML/Markdown reports from the completed diagnostic outputs."""
from pathlib import Path
import json
import html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results'

def markdown_table(df):
    d=df.copy()
    for c in d:
        if pd.api.types.is_float_dtype(d[c]):
            d[c]=d[c].map(lambda x: '—' if pd.isna(x) else f'{x:.5f}')
    return '| '+' | '.join(map(str,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in d.itertuples(index=False,name=None))

def main():
    o=json.loads((OUT/'overview.json').read_text())
    cov=pd.read_csv(OUT/'site_coverage.csv')
    gap=pd.read_csv(OUT/'missing_summary.csv')
    runs=pd.read_csv(OUT/'missing_runs.csv')
    categories=runs.groupby('category').length_hours.agg(['count','sum','median','max']).reindex(['leading_long','1-3h','4-24h','>24h'])
    categories[['count','sum']]=categories[['count','sum']].fillna(0).astype(int)
    categories.to_csv(OUT/'missing_category_counts.csv')
    lengths=pd.read_csv(OUT/'missing_length_distribution.csv')
    bench=pd.read_csv(OUT/'benchmark_summary.csv')
    ci=pd.read_csv(OUT/'benchmark_mae_site_bootstrap_ci.csv')
    bench=bench.merge(ci,on=['scenario','category','method','scope'])
    bysite=pd.read_csv(OUT/'cf_by_site.csv')
    monthly=pd.read_csv(OUT/'cf_by_month.csv')
    hour=pd.read_csv(OUT/'cf_by_hour.csv')
    fig,axes=plt.subplots(2,1,figsize=(14,8))
    x=np.arange(len(bysite))
    axes[0].fill_between(x,bysite.p05,bysite.p95,alpha=.2,label='p05–p95')
    axes[0].plot(x,bysite['mean'],label='Mean CF'); axes[0].plot(x,bysite.p50,label='Median CF')
    axes[0].set(xlabel='Site order PV001–PV253',ylabel='Observed CF'); axes[0].legend()
    x=np.arange(len(monthly))
    axes[1].fill_between(x,monthly.p05,monthly.p95,alpha=.2,label='p05–p95')
    axes[1].plot(x,monthly['mean'],label='Mean CF'); axes[1].plot(x,monthly.p50,label='Median CF')
    axes[1].set_xticks(x[::3],monthly.month.iloc[::3],rotation=45)
    axes[1].set(xlabel='Calendar month',ylabel='Observed CF'); axes[1].legend()
    plt.tight_layout(); plt.savefig(OUT/'cf_site_month_distribution.png',dpi=140); plt.close()
    tables=[]
    tables.append(('결측구간 분류 (시작부 장기결측 우선 분류)',categories.reset_index()))
    tables.append(('결측 위치별 집계',gap))
    tables.append(('연속 결측 길이 분포 (시간)',lengths[['position','count','mean','min','50%','75%','90%','95%','99%','max']]))
    selected=['category','method','restored_hours','coverage','mae_cf','rmse_cf','bias_cf','mae_ci_low','mae_ci_high']
    tables.append(('내부 결측: 주간, 방법별 복원 범위',bench[(bench.scenario=='internal')&(bench.scope=='daylight')][selected]))
    tables.append(('내부 결측: 네 방법의 공통 주간 시각',bench[(bench.scenario=='internal')&(bench.scope=='common_daylight')][selected]))
    tables.append(('시작부 모사: 주간',bench[(bench.scenario=='leading')&(bench.scope=='daylight')][selected]))
    tables.append(('관측률 하위 10개 발전소',cov.sort_values('observation_rate').head(10)[['site_id','first_observed','last_observed','observation_rate','active_span_observation_rate','leading_missing_hours','internal_missing_hours','trailing_missing_hours']]))
    peak=hour.loc[hour['mean'].idxmax()]
    lead=o['leading_missing_hours']/o['missing_hours']
    paragraphs=[
        f"학습 기준 시간축은 train/smp.csv의 {o['training_start']} ~ {o['training_end']}이며, 발전소 {o['sites']}개 × {o['expected_hours_per_site']:,}시간 = {o['expected_cells']:,}개의 예상 관측입니다. 실제 발전량은 {o['observed_rows']:,}행으로 전체 관측률은 {o['observation_rate']:.2%}입니다. 첫·마지막 관측 사이를 합산한 관측률은 {o['pooled_active_observation_rate']:.2%}입니다.",
        f"전체 결측 {o['missing_hours']:,}시간 중 시작부는 {o['leading_missing_hours']:,}시간({lead:.2%}), 내부는 {o['internal_missing_hours']:,}시간, 종료부는 {o['trailing_missing_hours']:,}시간입니다. 시작부 공백은 편입 이전일 가능성이 있어 일반 통신 결측과 구분해야 합니다. 시작부라는 위치만으로 원인은 확정할 수 없습니다.",
        "실제 내부 결측은 PV051의 2021-06-15 18:00 ~ 2021-06-16 00:00 KST 7시간 한 구간입니다. 1~3시간 결측은 0건이고 시작부를 제외한 24시간 초과 결측도 0건입니다. 분류 경계 중복을 없애기 위해 정확히 24시간은 4~24시간에 포함하고 장기는 25시간 이상으로 정의했습니다. 시작부 장기결측은 24시간 초과인 첫 관측 이전 공백에 우선 적용했습니다. 253개소의 첫 관측은 2021-01-19 13:00부터 2024-12-24 15:00까지 분산되어 있으며 마지막 관측은 모두 2025-06-01 00:00입니다.",
        f"시간별 CF = energy(kWh) / capacity_kw(kW) / 1시간입니다. 값을 자르거나 보정하지 않았습니다. CF > 1은 {o['quality']['cf_above_1']:,}건, CF > 1.2는 {o['quality']['cf_above_1_2']:,}건, 음수 발전량은 {o['quality']['negative_energy']:,}건이며 최대 CF는 {o['quality']['max_cf']:.4f}입니다. 관측 기준 평균 CF가 가장 높은 시각은 KST 구간 종료 {int(peak.hour)}시({peak['mean']:.4f})입니다. 야간과 발전소 구성 변화가 월별 분포에 영향을 주므로 월평균을 효율 변화로 단정하지 않습니다.",
        f"Artificial missing 실험은 seed={o['seed']}, 총 {o['artificial_blocks']:,}블록, {o['artificial_target_hours']:,}평가 시간입니다. 발전소별 각 길이마다 최대 2개 블록을 균일 무작위 추출했습니다. 내부 길이는 1·2·3·6·12·24·48·72·168시간, 시작부 모사는 168·720시간입니다. 후보는 CF가 [0,1.2] 안에 있고 전 구간이 관측되며 주간을 포함합니다. 내부 결측은 양끝 관측값도 요구합니다. 실험 블록끼리 겹칠 수 있으며 각 실험을 독립적으로 수행합니다.",
        "각 실험에서 대상 발전소 블록의 발전량을 완전히 숨깁니다. 시작부 실험은 블록 이전의 해당 발전소 관측도 전부 접근 불가로 처리하여 과거 관측이 없는 상태를 모사합니다. 실제 편입 전에는 정답이 없어 직접 성능을 검증할 수 없으므로 이 결과는 모사 실험입니다. Linear는 양끝 관측 사이 직선으로, 시작부 외삽은 하지 않습니다. Same-site same-hour는 ±30일 같은 KST 시간의 관측 평균입니다. Weather-similar day는 A·B 실측 운량·기온·습도·강수량의 07~18시 일평균을 표준화하고 연중 일자 ±45일 후보 중 가장 가까운 최대 5일의 같은 시각 CF 평균입니다. 일평균 변수마다 최소 8시간 기상 관측, 거리 계산마다 최소 6/8개 공유 변수를 요구하며 결측 블록이 속한 날짜는 모두 후보에서 제외합니다. Same-weather-station neighbor CF는 A 또는 B 지점을 공유하는 타 발전소의 동시각 관측 CF 평균입니다. 임의 대체 방법이나 야간 강제 0은 사용하지 않았습니다.",
        "평가값은 시간별 MAE·RMSE·bias(CF 단위), MAE(kWh), 복원률입니다. 0.01 CF 오차는 설비용량의 1%에 해당하는 1시간 평균 출력 오차입니다. 주간은 우측닫힘 구간의 중앙 시각(-30분)에서 NOAA 근사 태양고도가 0도보다 큰 시간으로 정의했습니다. 방법별 오차는 실제 복원 가능한 시각만의 결과이므로 공통 시각 표를 함께 봐야 합니다. common scope의 coverage 분모는 전체 결측 시간이 아닌 공통 시각 수입니다. 오차 신뢰구간은 발전소 단위 bootstrap 1,000회, 95%이며 시간 가중 평균을 사용합니다.",
        "이 실험은 미래 관측과 실측 기상을 활용하는 사후 복원 기준입니다. 이웃 방법은 다른 발전소가 같은 시각에 관측된 단일 발전소 결측 상황을 가정합니다. 동시 다발 결측·계절별 검증·실제 긴 시작부 공백의 복원 가능성을 보장하지 않으며, 표본 최대 길이를 넘는 장기결측에도 그대로 적용할 수 없습니다. 기상 변수 범위를 벗어난 값은 유사도 계산에서만 제외했으며 원본은 수정하지 않았습니다. A 실측 기상에 중복 키 초과 행이 6개 있지만 전부 07~18시 밖이므로 유사일 특징 계산에 사용되지 않습니다. 사용 시간대의 A·B 기상 키는 유일함을 검증했습니다.",
        "원본 5개 입력 파일의 SHA-256을 실행 전후 비교해 동일함을 확인했습니다. observed_capacity_factor.parquet는 원본 관측 행의 CF만 담은 파생 분석 파일로, 결측값을 채운 발전량 파일은 생성하지 않았습니다. 발전소별 시작·종료 시점과 전체/활성 구간 관측률은 site_coverage.csv, 모든 결측구간은 missing_runs.csv에 있습니다."
    ]
    findings=[]
    comparison=bench[(bench.scenario=='internal')&(bench.scope=='common_daylight')]
    for cat in ['1-3h','4-24h','>24h']:
        d=comparison[comparison.category==cat].dropna(subset=['mae_cf'])
        if len(d):
            best=d.loc[d.mae_cf.idxmin()]
            findings.append(f"{cat}: {best.method}의 주간 공통 시각 MAE가 가장 낮음({best.mae_cf:.5f} CF)")
    paragraphs.insert(6,'내부 결측 비교 결과 — '+'; '.join(findings)+'. 순위는 이 artificial missing 표본과 설정에 대한 결과입니다.')
    figs=[('coverage','발전소별 관측률'),('coverage_month_heatmap','발전소·월별 관측률'),('missing_runs','결측 길이와 결측 시간'),
        ('cf_distributions','CF 전체·월·시간 분포'),('cf_site_month_distribution','발전소별·연월별 CF 분위수'),
        ('cf_site_hour','발전소·시간별 평균 CF'),('cf_month_hour','월·시간별 평균 CF'),
        ('benchmark_errors','내부 결측 복원오차'),('benchmark_coverage_common','복원률 및 공통 주간 비교'),('benchmark_leading','시작부 모사 복원오차')]
    title='발전량 관측·CF·결측 복원 비교 분석'
    md='# '+title+'\n\n'+'\n\n'.join(paragraphs)+'\n\n'
    for heading,table in tables:
        md+='## '+heading+'\n\n'+markdown_table(table)+'\n\n'
    for name,caption in figs:
        md+='## '+caption+'\n\n!['+caption+']('+name+'.png)\n\n'
    (OUT/'report.md').write_text(md,encoding='utf-8')
    body='<h1>'+title+'</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
    for heading,table in tables:
        body+='<h2>'+heading+'</h2><div class="table">'+table.to_html(index=False,float_format=lambda x:f'{x:.5f}',na_rep='—')+'</div>'
    for name,caption in figs:
        body+='<h2>'+caption+'</h2><img src="'+name+'.png" alt="'+caption+'">'
    body+='<h2>전체 결과 파일</h2><ul>'+''.join('<li><a href="'+p.name+'">'+p.name+'</a></li>' for p in sorted(OUT.iterdir()) if p.suffix in ['.csv','.json','.parquet'])+'</ul>'
    (OUT/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+title+'</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.7;color:#182638}img{max-width:100%;height:auto}.table{overflow:auto}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #dde4ec;padding:8px;white-space:nowrap}h2{margin-top:48px}p{max-width:1000px}</style>'+body+'</html>',encoding='utf-8')
    print('Report saved:',OUT/'report.html')

if __name__=='__main__':main()
