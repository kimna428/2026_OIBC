"""Lazy day features; forecast fusion never consumes targets or observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.deep_learning.features import day_hours,aware,interval_max
from .sj_weather import fuse_series,make_features
from .sj_scaling import transform_rule,feature_transform

FC=['wa_cloud','wa_temp','wa_humidity','wa_rain','wa_wind_speed','wa_ground_press',
    'wb_cloud','wb_temp','wb_rel_hum','wb_uv_idx','wb_precip_prob','wb_total_liq','wb_vis','wb_wind_speed','wb_dew_point']
OBS_B={'wb_cloud':'wb_cloud','wb_temp':'wb_temp','wb_rel_hum':'wb_rel_hum','wb_uv_idx':'wb_uv_idx',
       'wb_vis':'wb_vis','wb_wind_speed':'wb_wind_speed','wb_dew_point':'wb_dew_point','wb_total_liq':'wb_precip_1h'}
AUX_ALL=[i for i,c in enumerate(FC) if c!='wb_precip_prob']


def context(frame,key,variables):
    frame=frame.sort_values([key,'ts']).reset_index(drop=True).copy()
    frame['_day']=(frame.ts-pd.Timedelta(hours=1)).dt.normalize()
    groups=frame.groupby([key,'_day'],observed=True)[variables]
    out=frame[[key,'ts']].copy()
    for suffix,block in [('previous_hour',groups.shift(1)),('next_hour',groups.shift(-1)),('day_mean',groups.transform('mean')),('day_std',groups.transform('std')),('day_max',groups.transform('max'))]:
        for c in variables:out[c+'__'+suffix]=block[c].to_numpy('float32')
    return out


def cloud_trends(a,b):
    # SJ temporal.py's exact 13 columns at ending hours11..15.
    out=np.full((len(a),24,13),np.nan,dtype='float32');phase=np.arange(1,6,dtype='float32')/6
    aa,az=a[:,9],a[:,15];peak=b[:,10:15];mean=peak.mean(1);early=peak[:,:2].mean(1);late=peak[:,2:].mean(1)
    vals=[np.broadcast_to(aa[:,None],peak.shape),np.broadcast_to(az[:,None],peak.shape),np.broadcast_to((az-aa)[:,None],peak.shape),np.broadcast_to(phase,peak.shape),
          a[:,10:15]-((1-phase)*aa[:,None]+phase*az[:,None]),np.broadcast_to(mean[:,None],peak.shape),peak-mean[:,None],np.broadcast_to((late-early)[:,None],peak.shape),
          np.where(np.arange(5)[None,:]<2,peak,np.nan),np.where(np.arange(5)[None,:]>=2,peak,np.nan),np.broadcast_to(np.isfinite(peak).sum(1)[:,None],peak.shape),
          np.broadcast_to(((az-aa)/6-(b[:,14]-b[:,10])/4)[:,None],peak.shape),np.broadcast_to((np.sign(az-aa)*np.sign(b[:,14]-b[:,10]))[:,None],peak.shape)]
    out[:,10:15]=np.stack(vals,-1)
    return out


class Features:
    def __init__(self,cache,sj_spec):
        self.cache=cache;self.spec=sj_spec;self.names=cache.manifest['weather_features'];self._day=None
        self.fused={};self._station_frames={};self.teacher=None
    def raw(self,d,name,round=17):
        return np.asarray(self.cache.weather[d])[...,self.names.index(name if round==17 else name+'_r10')]
    def build_sj_weather(self):
        if self.fused:return
        c=self.cache
        base=[n for n in self.names if not n.endswith(('_r10','_revision'))]
        for provider in ('wa','wb'):
            key=provider+'_station';columns=[n for n in base if n.startswith(provider+'_')]
            stations=sorted(c.sites[key].dropna().unique());arrays={}
            # Monthly mmap reads; copy each provider only once, no site-wide feature cache.
            positions=[int(np.flatnonzero(c.sites[key].to_numpy()==station)[0]) for station in stations]
            v17=np.stack([np.asarray(c.weather[d,positions])[...,[self.names.index(n) for n in columns]] for d in range(len(c.days))])
            v10=np.stack([np.asarray(c.weather[d,positions])[...,[self.names.index(n+'_r10') for n in columns]] for d in range(len(c.days))])
            for j,station in enumerate(stations):
                values={};times=pd.DatetimeIndex([h for day in c.days for h in day_hours(day)])
                for k,n in enumerate(columns):
                    a,b=v17[:,j,:,k].ravel().astype('float64'),v10[:,j,:,k].ravel().astype('float64')
                    # Same physically invalid filtering and WB wind units as reviewed SJ.
                    for z in (a,b):
                        suffix=n.removeprefix(provider+'_')
                        bad=~np.isfinite(z)
                        if suffix in ('wind_speed','wind_gust_spd','rain','snow','ice','total_liq','vis','ceiling','uv_idx'):bad|=z<0
                        if suffix=='wind_dir':bad|=(z<0)|(z>360)
                        z[bad]=np.nan
                        if n in ('wb_wind_speed','wb_wind_gust_spd'):z/=3.6
                    val,source,age,missing=fuse_series(a,b,circular=n.endswith('_wind_dir'))
                    values[n]=val;values[n+'__source']=source;values[n+'__missing17']=missing.astype('uint8');values[n+'__age_days']=age
                arrays[station]=pd.DataFrame({key:station,'ts':times,**values})
            self.fused[provider]=arrays
    def sj_frames(self,d):
        self.build_sj_weather();start=d*24;end=start+24
        return {p:pd.concat([f.iloc[start:end] for f in self.fused[p].values()],ignore_index=True) for p in ('wa','wb')}
    def sj(self,d):
        if self._day is not None and self._day[0]==d:return self._day[1]
        c=self.cache;frames=self.sj_frames(d);times=day_hours(c.days[d])
        keys=pd.DataFrame(dict(site_id=np.repeat(c.sites.site_id.to_numpy(),24),ts=np.tile(times,c.num_sites)))
        f,_,sun,_=make_features(keys,c.sites,frames['wa'],frames['wb'])
        for p,cols in [('wa',['wa_cloud','wa_temp']),('wb',['wb_cloud','wb_uv_idx'])]:
            ctx=context(frames[p],p+'_station',cols).set_index([p+'_station','ts'])
            ix=pd.MultiIndex.from_arrays([keys.site_id.map(c.sites.set_index('site_id')[p+'_station']),keys.ts])
            f=pd.concat([f,ctx.reindex(ix).reset_index(drop=True)],axis=1)
        names=self.spec['columns'];raw=f[names].to_numpy('float32').reshape(c.num_sites,24,len(names))
        wa=frames['wa'].set_index(['wa_station','ts']).wa_cloud
        wb=frames['wb'].set_index(['wb_station','ts']).wb_cloud
        get=lambda p,s: s.reindex(pd.MultiIndex.from_arrays([keys.site_id.map(c.sites.set_index('site_id')[p+'_station']),keys.ts])).to_numpy().reshape(c.num_sites,24)
        extra=cloud_trends(get('wa',wa),get('wb',wb))
        mean=np.full((c.num_sites,24,2),np.nan,dtype='float32');mean[...,1]=0
        if self.teacher is not None:
            values=self.teacher[d]
            mean[:,10:15,0]=values[:,None];mean[:,10:15,1]=np.isfinite(values)[:,None]
        xgb=np.concatenate([raw,extra,mean],axis=-1)
        result=dict(raw=raw,xgb=xgb,day=sun.reshape(c.num_sites,24))
        self._day=(d,result);return result
    def state(self,d):
        c=self.cache;weather=np.asarray(c.weather[d]);parts=[]
        for r in (10,17):
            x=weather[...,[self.names.index(n if r==17 else n+'_r10') for n in FC]]
            miss=~np.isfinite(x[...,[0,6]])
            parts.extend([x,miss.astype('float32')])
        times=day_hours(c.days[d]);repeated=times.repeat(c.num_sites)
        # Same NOAA geometry as SJ/01_state; 01_state subtracts30min in solar.
        from .state_geometry import solar
        lat=np.tile(c.sites.lat,len(times));lon=np.tile(c.sites.lon,len(times))
        mid=repeated-pd.Timedelta(minutes=30)
        elev,az,cosz=solar(repeated,lat,lon)
        elev=elev.reshape(24,c.num_sites).T;az=az.reshape(24,c.num_sites).T;cosz=cosz.reshape(24,c.num_sites).T
        # Use common geometric definition consistently; explicitly a corrected variant.
        doy=times.dayofyear.to_numpy();hour=np.arange(24)
        geo=np.stack([elev/90,cosz,np.sin(np.deg2rad(az)),np.cos(np.deg2rad(az)),
                      np.broadcast_to(np.sin(2*np.pi*doy/365.25),elev.shape),np.broadcast_to(np.cos(2*np.pi*doy/365.25),elev.shape)],-1)
        st=np.stack([np.log(c.sites.capacity_kw),c.sites.lat,c.sites.lon],-1)
        st=(st-st.mean(0))/(st.std(0)+1e-6)
        hh=np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)],-1)
        raw=np.concatenate([*parts,geo,np.broadcast_to(hh,(c.num_sites,24,2)),np.broadcast_to(st[:,None,:],(c.num_sites,24,3))],-1).astype('float32')
        return dict(raw=raw,day=elev>0,elevation=elev,azimuth=az,cosz=cosz)


def generation_frame(history):
    rows=[]
    for d,day in enumerate(history.days):
        actual=np.asarray(history.energy[d]);s,h=np.where(np.isfinite(actual))
        rows.append(pd.DataFrame(dict(site_id=history.sites.site_id.to_numpy()[s],ts=day_hours(day)[h],energy=actual[s,h])))
    return pd.concat(rows,ignore_index=True)


def outages(history,minimum=3):
    # Operates ONLY on a cutoff-masked History, including boundary qualification.
    g=generation_frame(history);g['date']=g.ts.dt.normalize();g['hour']=g.ts.dt.hour
    d=g.loc[g.hour.between(8,17)].groupby(['site_id','date']).energy.agg(['max','size']).reset_index()
    d=d.loc[d['size']>=8];active=d.groupby('date').site_id.nunique();low=d.loc[d['max']<1]
    share=low.groupby('date').site_id.nunique()/active;low=low.loc[~low.date.isin(share[share>=.2].index)].sort_values(['site_id','date'])
    run=low.date.groupby(low.site_id).diff().ne(pd.Timedelta(days=1)).cumsum()
    return low.loc[run.map(run.value_counts()).to_numpy()>=minimum,['site_id','date']]


def observed_targets(root,history,features):
    """Clean/fit auxiliary labels using ONLY observations and forecasts before cutoff."""
    result={};bounds={};c=history.base
    train_days=[int(i) for i in history.indices]
    low=np.full(len(FC),np.inf);high=np.full(len(FC),-np.inf)
    for d in train_days:
        raw=np.asarray(c.weather[d])
        for r in (10,17):
            v=raw[...,[features.names.index(n if r==17 else n+'_r10') for n in FC]]
            low=np.minimum(low,np.where(np.isfinite(v),v,np.inf).min((0,1)))
            high=np.maximum(high,np.where(np.isfinite(v),v,-np.inf).max((0,1)))
    for provider,cols,mapping in [('wa',FC[:6],{n:n for n in FC[:6]}),('wb',list(OBS_B),OBS_B)]:
        columns=[provider+'_station','ts']+list(mapping.values())
        o=pd.read_parquet(Path(root)/f'train/observed_{provider}.parquet',columns=columns,
             filters=[('ts','<=',history.cutoff.to_pydatetime()),('ts','>=',day_hours(history.days[0])[0].to_pydatetime())])
        o=o.rename(columns={v:k for k,v in mapping.items()});o['ts']=pd.to_datetime(o.ts).dt.tz_convert('Asia/Seoul')
        o=o.drop_duplicates([provider+'_station','ts'])
        for n in cols:
            lo,hi=low[FC.index(n)],high[FC.index(n)]
            bounds[n]=[lo,hi] if np.isfinite(lo) else [None,None]
            o.loc[~np.isfinite(o[n])|(o[n]<lo-.1*(hi-lo))|(o[n]>hi+.1*(hi-lo)),n]=np.nan
        result[provider]=o.set_index([provider+'_station','ts'])
    return result,bounds


def auxiliary_day(obs,cache,d):
    times=day_hours(cache.days[d]);out=np.full((cache.num_sites,24,len(FC)),np.nan,dtype='float32')
    for j,n in enumerate(FC):
        p=n[:2]
        if n not in obs[p]:continue
        ix=pd.MultiIndex.from_arrays([np.repeat(cache.sites[p+'_station'],24),np.tile(times,cache.num_sites)])
        out[...,j]=obs[p][n].reindex(ix).to_numpy().reshape(cache.num_sites,24)
    return out


def sj_zero_spells(history):
    g=generation_frame(history);g['date']=(g.ts-pd.Timedelta(hours=1)).dt.normalize()
    daily=g.groupby(['site_id','date']).energy.agg(['size',lambda x:bool((x==0).all())]).reset_index()
    daily.columns=['site_id','date','size','zero'];daily=daily.loc[(daily['size']==24)&daily.zero].sort_values(['site_id','date'])
    # At least four high-sun hours, based only on metadata and timestamps.
    from .sj_weather import solar_position
    good=[]
    meta=history.sites.set_index('site_id')
    for row in daily.itertuples():
        t=day_hours(row.date)-pd.Timedelta(minutes=30)
        elev=solar_position(t,np.repeat(meta.loc[row.site_id,'lat'],24),np.repeat(meta.loc[row.site_id,'lon'],24))[0]
        good.append(int((elev>15).sum())>=4)
    daily=daily.loc[good]
    runs=daily.date.groupby(daily.site_id).diff().ne(pd.Timedelta(days=1)).cumsum()
    result=daily.loc[runs.map(runs.value_counts()).to_numpy()>=7,['site_id','date']].copy()
    result.attrs['operating_day']=True
    return result
