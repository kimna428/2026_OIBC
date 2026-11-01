"""Build a results-focused three-page main report without changing the technical appendix."""
from pathlib import Path
import hashlib
import html
import json
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/third_experiment'
OLD=ROOT/'reports/expanding'

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def one(frame,**keys):
    for key,value in keys.items(): frame=frame[frame[key]==value]
    if len(frame)!=1: raise ValueError(f'Expected one result: {keys}, found {len(frame)}')
    return frame.iloc[0]
def won(x): return f'{x:,.0f}'
def number(x): return f'{x:,.2f}'
def percent(before,after): return 100*(before-after)/before

def inline(text):
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
    return re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',text)

def section_html(md):
    lines=md.strip().splitlines(); result=[]; i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1; continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'))
            result.append(f'<h{level}>{inline(line[level:].strip())}</h{level}>'); i+=1
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells): table.append(cells)
                i+=1
            result.append('<table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in table[0])+'</tr></thead><tbody>')
            for row in table[1:]: result.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
            result.append('</tbody></table>')
        elif line.startswith('- '):
            result.append('<ul>')
            while i<len(lines) and lines[i].strip().startswith('- '):
                result.append('<li>'+inline(lines[i].strip()[2:])+'</li>'); i+=1
            result.append('</ul>')
        else:
            paragraph=[]
            while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','- ')):
                paragraph.append(lines[i].strip()); i+=1
            result.append('<p>'+inline(' '.join(paragraph))+'</p>')
    return '\n'.join(result)

def main():
    appendix=OUT/'report.md'; before=digest(appendix)
    metric=pd.read_csv(OUT/'metrics_holdout.csv')
    scores=pd.read_csv(OUT/'selected_composition_metrics.csv')
    old_metric=pd.read_csv(OLD/'metrics.csv')
    old_score=pd.read_csv(OLD/'official_formula_scores.csv')
    lock=json.loads((OUT/'selection_lock.json').read_text())
    new={}; old={}
    for family in ['B','C']:
        choice=lock['chosen'][family]
        new[family]={'metric':one(metric,candidate=choice['candidate'],policy=choice['policy'],fold=5,dimension='overall'),'balanced':one(scores,family=family,fold=5,composition='balanced',group='total'),'optimized':one(scores,family=family,fold=5,composition='optimized',group='total')}
        old[family]={'metric':one(old_metric,experiment='V3_'+family,fold=5,dimension='overall'),'score':one(old_score,experiment='V3_'+family,fold=5)}
    b=new['B']; c=new['C']; ob=old['B']; oc=old['C']
    offline=one(pd.read_csv(OUT/'offline_C_metrics.csv'),fold=5,dimension='overall',policy=lock['chosen']['C']['policy'])
    opt=pd.read_csv(OUT/'optimization_development.csv')
    dev_base=one(opt,family='B',composition='balanced'); dev_opt=one(opt,family='B',composition='optimized')
    sites=pd.read_csv(ROOT/'handover 2/01_participant_dataset/sites.csv')
    def capacities(path):
        return pd.read_csv(path).merge(sites[['site_id','capacity_kw']],on='site_id',validate='one_to_one').groupby('group').capacity_kw.sum()
    balanced_cap=capacities(OLD/'composition.csv'); optimized_cap=capacities(OUT/'composition_B.csv')
    night=one(metric,candidate=lock['chosen']['B']['candidate'],fold=5,policy='train_calibrated',dimension='period',label='twilight')
    night_raw=one(metric,candidate=lock['chosen']['B']['candidate'],fold=5,policy='raw',dimension='period',label='twilight')
    table=[]
    for label,m,s in [('2차 B',ob['metric'],ob['score']),('3차 선정 B',b['metric'],b['balanced']),('2차 C',oc['metric'],oc['score']),('3차 선정 C',c['metric'],c['balanced'])]:
        kwh=m['mae_kwh'] if 'mae_kwh' in m.index else m['energy_mae_kwh']
        table.append(f"| {label} | {m.cf_mae:.5f} | {kwh:.3f} | {won(s.penalty_krw)} | {won(s.score_krw)} |")
    text=f'''# OIBC 2026 — 3차 모델 개선 결과

발전량 예측 정확도와 집합자원 정산 성과를 함께 개선하기 위한 실험

## 1. 실험 목적과 핵심 성과

2차 실험에서는 기상예보와 태양고도를 사용하는 B 모델, 여기에 과거 발전량을 추가한 C 모델이 유망했다. 3차 실험의 목적은 두 모델의 검증 신뢰성을 확인하고, 학습량 확대·예보 활용·야간 처리가 실제 정산 성과를 높이는지 평가하는 것이었다. 발전소별 오차를 줄이는 것과 두 집합의 정산 페널티를 줄이는 것이 같은 결과를 내는지도 살펴보았다.

**현재 추천 후보는 100만 행을 학습하고, 전일 10시·17시 예보와 두 예보의 수정량을 함께 사용하는 B 모델이다.** 후보와 후처리는 개발 구간에서 먼저 선택한 뒤 2025년 3~5월에 평가했다. 같은 G1/G2 배정을 유지한 비교에서 B의 CF MAE는 {ob['metric'].cf_mae:.5f}에서 {b['metric'].cf_mae:.5f}로 **{percent(ob['metric'].cf_mae,b['metric'].cf_mae):.2f}% 감소**했고, 공식 페널티는 **{percent(ob['score'].penalty_krw,b['balanced'].penalty_krw):.2f}% 감소**했다. OOF 예측으로 고른 그룹 배정을 적용하면 페널티가 추가로 {percent(b['balanced'].penalty_krw,b['optimized'].penalty_krw):.2f}% 줄었다.

개선 폭의 대부분은 예측 모델과 후처리 변경에서 나왔다. 그룹 배정도 봄 평가에서 개선을 보였지만 추가 효과는 작았다. 따라서 이번 결과는 예측 개선의 효과와 그룹 최적화의 효과를 구분해 해석할 필요가 있다.

## 2. 2차 대비 변경한 모델 및 전처리

기존의 CF 정규화와 음수·CF 초과 관측의 학습 제외 방식을 유지하면서, 학습 데이터 규모를 20만·100만·전체 행으로 비교했다. 트리 수와 학습률 등도 시간순 검증으로 조정했다. 전체 행을 사용한 모델이 항상 공식 페널티에서 가장 좋지는 않았으며, 개발 구간에서는 100만 행을 사용한 B 후보가 선정됐다. 학습량 확대의 효과는 예보 종류를 고정한 비교에서 확인했다.

예보는 기존의 17시 라운드 외에 10시 라운드, 두 라운드의 결합, 두 예보 사이의 수정량을 비교했다. 10시 단독 예보보다 17시 예보가 유리했고, 두 시점의 정보를 함께 활용한 B 후보가 개발 구간의 공식 페널티에서 가장 좋은 결과를 보였다. 예보 수정량은 대상일의 기상 전망이 어떻게 바뀌었는지를 모델에 전달하는 정보로 해석할 수 있다. 다만 결합·수정량의 추가 효과가 모든 fold에서 일관되게 나타난 것은 아니다.

야간에는 원래 예측 유지, 완전한 야간에만 0 처리, 학습 실적으로 발전소별 임계값을 정하는 세 방법을 비교했다. 야간 처리는 불필요한 양의 예측을 줄였고 학습 기반 규칙이 선택됐다. 봄의 일출·일몰 인접 구간 CF MAE는 {night_raw.cf_mae:.5f}에서 {night.cf_mae:.5f}로 낮아졌지만, 실제 발전량 약 {night.removed_actual_kwh:.1f} kWh도 0 처리됐다. 오차 개선과 실제 발전량 제거가 함께 발생한다는 점을 확인했다.

검증 시작의 00시 경계는 기존 코드가 전날 운영일로 정상 처리하고 있어 샘플 변경은 없었다. 추가로 하이퍼파라미터 선택에도 정보 마감을 적용했고, 봄 첫 발행 시각 이후인 개발 마지막 7시간은 모델·그룹 선택에서 제외했다. 원본 데이터는 변경하지 않았다.

<!-- page -->

## 3. B/C 성능과 정산 페널티 비교

다음은 **2025년 3~5월의 동일한 2,208시간과 동일한 균형 배정**에서 비교한 결과다. 3차 B/C는 각각 개발 구간에서 선정한 후보이며, 봄 성능을 보고 다시 고른 모델이 아니다. CF MAE와 개별 MAE는 발전소 관측 단위의 평균이다. 금액은 G1/G2 합계이며 원 단위로 반올림했다. 정산 값은 제공 README 산식을 적용한 로컬 검증 값이다.

| 모델 | CF MAE | 개별 MAE(kWh) | 공식 페널티(원) | 공식 정산금(원) |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

B와 C 모두 2차 대비 발전소별 오차와 공식 페널티를 줄였다. B의 페널티 감소액은 약 {(ob['score'].penalty_krw-b['balanced'].penalty_krw)/10000:,.1f}만 원, C는 약 {(oc['score'].penalty_krw-c['balanced'].penalty_krw)/10000:,.1f}만 원이다. 같은 평가 시각과 배정에서는 실적 수입이 같으므로, 페널티가 줄어든 만큼 공식 정산금이 증가한다.

**C는 발전소별 오차가 더 작았지만, 공식 페널티는 B가 더 낮았다.** 정산은 개별 발전소 오차의 절대값을 더하는 방식이 아니라, 각 집합의 발전량을 합산한 뒤 오차를 계산한다. 따라서 오차가 어느 시각에 발생했는지, 같은 집합 안에서 과대·과소예측이 얼마나 상쇄되는지가 중요하다. 설비용량과 SMP, 11~15시 가중치도 반영된다. CF MAE만으로 최종 후보를 고르면 정산 목표와 다른 선택을 할 수 있다는 점이 이번 실험에서도 확인됐다.

C에는 추가적인 사용 조건도 있다. 위 표의 C는 예측 기간에도 과거 발전 실적이 계속 수신된다고 가정한다. 제공 test 파일에는 이 실적 스트림이 없어, 첫 발행 이후 실적을 받지 않는 보조 실험을 수행했다. 이 경우 C의 봄 CF MAE는 {c['metric'].cf_mae:.5f}에서 **{offline.cf_mae:.5f}**로 악화됐다. C를 실제 제출에 활용하려면 실적 제공 여부와 수신 지연을 먼저 확인해야 한다. B는 제공 예보와 발전소 메타·시간 정보로 계산할 수 있어 현재 활용 조건이 더 명확하다.

학습 기반 야간 규칙은 보수적인 야간 규칙보다 공식 페널티를 조금 더 줄였지만, 추가 이득은 작았다. 이 규칙의 필요성은 발전량 제거 위험과 함께 판단해야 한다. 이번에 선택된 조합의 전체 개선을 개별 피처 하나의 효과로 단정하지는 않았다.

<!-- page -->

## 4. G1/G2 최적화 결과

그룹 배정은 봄 정답을 사용하지 않고, 이전 시점의 OOF 예측 오차만으로 결정했다. 원본 정답이 253개소 모두 존재하는 선택용 겨울 {int(dev_base.training_oof_hours):,}시간을 사용했으며, 발전소를 빠짐없이 두 집합에 배정하고 각 집합 용량을 20,000 kW 이상으로 유지했다. 아래는 추천 B 후보의 **봄 평가 결과**다.

| 배정 | G1 용량(kW) | G2 용량(kW) | 집합 MAE(kWh) | 공식 페널티(원) |
| --- | --- | --- | --- | --- |
| 기존 균형 배정 | {number(balanced_cap['G1'])} | {number(balanced_cap['G2'])} | {b['balanced'].mae_kwh:.2f} | {won(b['balanced'].penalty_krw)} |
| OOF 최적화 배정 | {number(optimized_cap['G1'])} | {number(optimized_cap['G2'])} | {b['optimized'].mae_kwh:.2f} | {won(b['optimized'].penalty_krw)} |

집합 MAE는 두 집합·동일 평가 시간의 평균이다. 최적화 배정은 용량 제약을 만족하면서 봄 페널티를 약 {(b['balanced'].penalty_krw-b['optimized'].penalty_krw)/10000:.1f}만 원({percent(b['balanced'].penalty_krw,b['optimized'].penalty_krw):.2f}%) 줄였다. 공식 정산금은 {won(b['optimized'].score_krw)}원으로 증가했다. 다만 선택용 겨울 OOF에서의 개선율 {percent(dev_base.penalty_krw,dev_opt.penalty_krw):.2f}%에 비하면 봄의 추가 이득은 작았다. 겨울에 유리한 오차 상쇄 구조가 봄에도 같은 크기로 유지되지는 않았다.

2차 B와 비교한 최종 조합의 봄 페널티 감소율은 {percent(ob['score'].penalty_krw,b['optimized'].penalty_krw):.2f}%다. 이 중 같은 배정에서의 모델 개선이 주된 성과이며, 그룹 최적화는 소폭의 추가 개선으로 보는 것이 적절하다. 서로 다른 fold의 총 정산금을 비교해 이 효과를 판단하지 않았다.

## 5. 한계 및 다음 실험 방향

이번 봄 구간은 2차 분석에서 이미 관찰한 자료다. 3차에서는 파라미터·후처리·그룹을 먼저 고정한 뒤 평가했지만, 완전히 새로운 미관측 검증이라고 할 수는 없다. 또한 전체 발전소의 정답이 동시에 존재하는 과거 구간이 짧아, 그룹 선택은 주로 겨울 자료에 의존했다. 정답이 없는 시각을 0으로 채워 평가 범위를 넓히지는 않았다.

3월에는 기상예보를 고려한 뒤에도 음의 잔차가 이어지는 기간이 있었지만, 다른 날짜에도 비슷하거나 더 큰 오차가 나타났다. 제공 자료만으로 산불 등 특정 원인을 확인할 수는 없었다. 이 분석을 근거로 특정 기간을 임의로 제외하거나 외부 산불 자료를 학습에 추가하지 않았다.

다음 실험은 세 방향을 우선한다.

- **선정 조합의 재검증:** B 후보를 고정하고 추가 연도·계절의 시간순 검증을 확대한다. 새 정답 구간이 확보되면 독립 평가를 수행한다.
- **정산 목표에 맞춘 학습:** 발전소 용량과 중요 시간대를 반영한 손실·오차 보정을 비교한다. C는 실적 수신 조건을 확인한 뒤 재평가한다.
- **배정 안정성 확인:** 다계절 OOF와 공통 관측 쌍의 잔차 통계를 활용해, 특정 겨울 구간에 의존하지 않는 배정을 검토한다.

현재는 B 선정 후보를 다음 실험의 기준으로 삼을 수 있다. 그룹 배정은 봄에서 추가 개선을 보였으나, 개선 폭이 작아 계절을 넘는 안정성을 더 확인해야 한다. 후보 모델은 봄 이전 자료로 학습한 검증용이며 최종 제출용 재학습 모델과는 구분한다.

기술 부록: [기존 report.md](report.md). 검증 경계, 모델 설정, 원시 성능표, 예보 변수별 중요도, 산불 잔차 분석과 재현 안내를 원문 그대로 보존했다. 실행 절차는 [analysis/README.md](../../analysis/README.md)에 정리돼 있다.
'''
    # Explicit three-page print layout; the Markdown remains convenient for repository review.
    pages=text.split('<!-- page -->')
    assert len(pages)==3
    css='''@page{size:A4;margin:0}*{box-sizing:border-box}body{margin:0;background:#e8edf2;color:#1b2531;font-family:"Apple SD Gothic Neo","Malgun Gothic","Noto Sans CJK KR",sans-serif;font-size:10pt;line-height:1.57;word-break:keep-all;overflow-wrap:break-word}.page{width:210mm;min-height:297mm;margin:10mm auto;padding:15mm 17mm 16mm;background:white;position:relative;box-shadow:0 2px 14px #0002;break-after:page}.page:last-child{break-after:auto}h1{font-size:20pt;line-height:1.25;letter-spacing:-.6px;margin:0 0 4mm;color:#123851}h2{font-size:13pt;line-height:1.3;margin:5mm 0 2.5mm;color:#123851;border-bottom:1px solid #bccbd5;padding-bottom:1.5mm}p{margin:0 0 3mm;text-align:justify}strong{font-weight:700;color:#123851}table{width:100%;border-collapse:collapse;table-layout:auto;font-size:8.8pt;margin:3mm 0 3mm;line-height:1.45;break-inside:avoid}th{background:#eaf0f5;color:#123851}th,td{padding:2mm 1.6mm;border-top:1px solid #cad5de;border-bottom:1px solid #cad5de;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}ul{padding-left:5mm;margin:2mm 0 3mm}li{margin:0 0 2mm}a{color:#245b80;text-decoration:none}.footer{position:absolute;bottom:7mm;left:17mm;right:17mm;font-size:8pt;color:#667684;border-top:1px solid #dde4e9;padding-top:2mm;display:flex;justify-content:space-between}@media print{body{background:white}.page{margin:0;box-shadow:none;width:210mm;height:297mm;min-height:0;overflow:visible}}@media screen and (max-width:820px){.page{width:100%;min-height:auto;padding:24px;margin:12px 0}.footer{position:static;margin-top:20px}}'''
    body=''.join(f'<article class="page">{section_html(page)}<footer class="footer"><span>OIBC 2026 · 3차 모델 개선 결과</span><span>{i} / 3</span></footer></article>' for i,page in enumerate(pages,1))
    (OUT/'main_report.md').write_text(text,encoding='utf-8')
    (OUT/'main_report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OIBC 2026 — 3차 모델 개선 결과</title><style>'+css+'</style></head><body>'+body+'</body></html>',encoding='utf-8')
    assert digest(appendix)==before,'Technical appendix was changed'
    inputs=[OUT/'metrics_holdout.csv',OUT/'selected_composition_metrics.csv',OUT/'selection_lock.json',OUT/'offline_C_metrics.csv',OUT/'optimization_development.csv',OUT/'composition_B.csv',OLD/'metrics.csv',OLD/'official_formula_scores.csv',OLD/'composition.csv',ROOT/'handover 2/01_participant_dataset/sites.csv']
    (OUT/'main_report_build.json').write_text(json.dumps({'technical_appendix':'report.md','appendix_sha256_before':before,'appendix_sha256_after':digest(appendix),'appendix_unchanged':True,'print_pages':3,'input_hashes':{str(p.relative_to(ROOT)):digest(p) for p in inputs},'outputs':['main_report.md','main_report.html']},indent=2),encoding='utf-8')
    print('Built main_report.md and main_report.html (3 print pages); technical appendix unchanged.')
    print('Characters per page:',[len(page) for page in pages])

if __name__=='__main__': main()
