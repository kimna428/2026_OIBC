"""Reviewed pure forecast functions derived from SJ weather.py. Original unchanged.
Source SHA256: cba4bf8432182a92e77ccde5c411eee08197adf67e0436ff2f2cfd0faf944c51
No teammate entry point is imported or executed.
"""
import numpy as np
import pandas as pd
SOURCE_NAMES={0:"unfilled",1:"round17",2:"round10",3:"interpolation",4:"previous_day"}

def fuse_series(v17, v10, *, circular=False, max_gap=3, hours_per_day=24):
    """Fuse complete target-day blocks. Never interpolate across issuance days.

    Previous-day lookup uses only the original 17/10 fused series; interpolation
    results and previously carried values never become historical sources.
    """
    a, b = np.asarray(v17, dtype='float64'), np.asarray(v10, dtype='float64')
    if len(a) != len(b) or len(a) % hours_per_day:
        raise ValueError('Expected equally sized complete target-day blocks')
    original = np.where(np.isfinite(a), a, b)
    if circular:
        original = original % 360
    out = original.copy()
    source = np.where(np.isfinite(a), 1, np.where(np.isfinite(b), 2, 0)).astype('uint8')
    age = np.zeros(len(a), dtype='float32')
    blocks = out.reshape(-1, hours_per_day)
    codes = source.reshape(-1, hours_per_day)
    for values, code in zip(blocks, codes):
        missing = ~np.isfinite(values)
        if not missing.any():
            continue
        delta = np.diff(np.r_[False, missing, False].astype('int8'))
        for left, right in zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1)):
            if left == 0 or right == hours_per_day or right-left > max_gap:
                continue
            f = np.arange(1, right-left+1)/(right-left+1)
            x, y = values[left-1], values[right]
            if circular:
                values[left:right] = (x + f*((y-x+180) % 360-180)) % 360
            else:
                values[left:right] = x+f*(y-x)
            code[left:right] = 3
    # Search the latest ORIGINAL day at each hour. No maximum age was requested.
    days = original.reshape(-1, hours_per_day)
    indices = np.arange(len(days))[:, None]
    last = np.maximum.accumulate(np.where(np.isfinite(days), indices, -1), axis=0)
    missing = ~np.isfinite(blocks)
    use = missing & (last >= 0)
    d, h = np.nonzero(use)
    blocks[d, h] = days[last[d, h], h]
    codes[d, h] = 4
    age.reshape(-1, hours_per_day)[d, h] = d-last[d, h]
    age[~np.isfinite(out)] = np.nan
    return out.astype('float32'), source, age, ~np.isfinite(a)

def prepare_weather(frame, provider, stations, times):
    key = provider+'_station'
    columns = [c for c in frame if c not in [key, 'round', 'ts']]
    if frame.duplicated([key, 'round', 'ts']).any():
        raise ValueError('Duplicate forecast keys')
    if set(frame['round'].unique()) != {10, 17}:
        raise ValueError('Unexpected forecast round')
    frame = frame.copy()
    audit = {'provider': provider, 'off_grid_rows': int((~frame.ts.isin(times)).sum()),
             'variables': {}, 'invalid_to_missing': {}}
    # Retain humidity >100 and other uncertain values; flag them later. Only
    # physically invalid negative speed/amount/distance and invalid angles go NaN.
    nonnegative = ['wind_speed', 'wind_gust_spd', 'rain', 'snow', 'ice', 'total_liq', 'vis', 'ceiling', 'uv_idx']
    for c in columns:
        bad = ~np.isfinite(frame[c]) & frame[c].notna()
        if c.removeprefix(provider+'_') in nonnegative:
            bad |= frame[c] < 0
        if c.endswith('_wind_dir'):
            bad |= (frame[c] < 0) | (frame[c] > 360)
        audit['invalid_to_missing'][c] = int(bad.sum())
        frame.loc[bad, c] = np.nan
    # Explicit unit conversion, including B gust speed.
    if provider == 'wb':
        for c in ['wb_wind_speed', 'wb_wind_gust_spd']:
            frame[c] = frame[c]/3.6
    pieces = []
    for station in sorted(stations):
        sub = frame.loc[frame[key] == station]
        r17 = sub.loc[sub['round'] == 17].set_index('ts')[columns].reindex(times)
        r10 = sub.loc[sub['round'] == 10].set_index('ts')[columns].reindex(times)
        out = pd.DataFrame({key: station, 'ts': times})
        for c in columns:
            values, source, age, missing = fuse_series(r17[c], r10[c], circular=c.endswith('_wind_dir'))
            out[c] = values
            out[c+'__source'] = source
            out[c+'__missing17'] = missing.astype('uint8')
            out[c+'__age_days'] = age
            counts = audit['variables'].setdefault(c, {n: 0 for n in SOURCE_NAMES.values()})
            for code, name in SOURCE_NAMES.items():
                counts[name] += int((source == code).sum())
            counts['max_age_days'] = max(counts.get('max_age_days', 0), float(np.nanmax(age, initial=0)))
        pieces.append(out)
    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated([key, 'ts']).any():
        raise AssertionError('Expanded weather keys')
    return result, audit

def solar_position(ts, lat, lon):
    """NOAA approximation at supplied instants (KST). Based on supplied baseline."""
    t = pd.DatetimeIndex(ts)
    hour = t.hour.to_numpy()+t.minute.to_numpy()/60
    year = 2*np.pi/365*(t.day_of_year.to_numpy()-1+(hour-12)/24)
    eq = 229.18*(.000075+.001868*np.cos(year)-.032077*np.sin(year)
                -.014615*np.cos(2*year)-.040849*np.sin(2*year))
    dec = (.006918-.399912*np.cos(year)+.070257*np.sin(year)
           -.006758*np.cos(2*year)+.000907*np.sin(2*year)
           -.002697*np.cos(3*year)+.00148*np.sin(3*year))
    ha = np.deg2rad((hour*60+eq+4*(lon-135))/4-180)
    la = np.deg2rad(lat)
    cosz = np.clip(np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha), -1, 1)
    elev = np.rad2deg(np.arcsin(cosz))
    az = np.rad2deg(np.arctan2(-np.sin(ha), np.tan(dec)*np.cos(la)-np.sin(la)*np.cos(ha))) % 360
    return elev.astype('float32'), az.astype('float32'), cosz.astype('float32')

def make_features(labels, sites, wa, wb):
    d = labels[['site_id', 'ts']].merge(sites.drop(columns='current_group'), on='site_id', validate='many_to_one', sort=False)
    for key, weather in [('wa_station', wa), ('wb_station', wb)]:
        d = d.merge(weather, on=[key, 'ts'], how='left', validate='many_to_one', sort=False)
    # Joins may promote timestamp resolution ms -> ns without changing instants.
    if not d.site_id.equals(labels.site_id) or not d.ts.eq(labels.ts).all():
        raise AssertionError('Changed label order')
    X = d.drop(columns=['ts', 'site_id', 'wa_station', 'wb_station']).copy()
    categorical = []
    for key in ['site_id', 'wa_station', 'wb_station']:
        # Categories come only from provided metadata, never from targets.
        X[key] = pd.Categorical(d[key], categories=sorted(sites[key].unique())).codes.astype('int16')
        categorical.append(key)
    for c in list(X.columns):
        if c.endswith('__source'):
            categorical.append(c)
        if c.endswith('_wind_dir'):
            theta = np.deg2rad(X.pop(c))
            X[c+'_sin'], X[c+'_cos'] = np.sin(theta), np.cos(theta)
        if c in ['wa_cloud', 'wa_humidity', 'wb_cloud', 'wb_rel_hum'] or c.endswith('_prob'):
            X[c+'__out_of_range'] = ((X[c]<0)|(X[c]>100)).astype('uint8')
    X['wb_temp_minus_dew_point'] = X.wb_temp-X.wb_dew_point
    X['wb_gust_minus_wind_speed'] = X.wb_wind_gust_spd-X.wb_wind_speed
    pairs = [('temp','temp'), ('cloud','cloud'), ('humidity','rel_hum'),
             ('wind_speed','wind_speed'), ('rain','rain'), ('snow','snow')]
    for a, b in pairs:
        # B snowfall cm -> mm for the cross-provider difference only.
        scale = 10 if a == 'snow' else 1
        X['ab_'+a+'_difference'] = X['wa_'+a]-X['wb_'+b]*scale
    # A source filled from old days is not evidence of provider disagreement.
    for a, b in pairs:
        ok = d['wa_'+a+'__source'].isin([1, 2]) & d['wb_'+b+'__source'].isin([1, 2])
        X.loc[~ok, 'ab_'+a+'_difference'] = np.nan
        X['ab_'+a+'__both_original'] = ok.astype('uint8')
    a = np.arctan2(X.wa_wind_dir_sin, X.wa_wind_dir_cos)
    b = np.arctan2(X.wb_wind_dir_sin, X.wb_wind_dir_cos)
    ok = d.wa_wind_dir__source.isin([1, 2]) & d.wb_wind_dir__source.isin([1, 2])
    X['ab_wind_dir_difference'] = np.where(ok, np.rad2deg(np.arctan2(np.sin(a-b), np.cos(a-b))), np.nan)
    X['ab_wind_dir__both_original'] = ok.astype('uint8')
    mid = pd.DatetimeIndex(d.ts)-pd.Timedelta(minutes=30)
    hour = mid.hour.to_numpy()+mid.minute.to_numpy()/60
    doy = mid.day_of_year.to_numpy()
    X['hour'] = hour
    X['hour_sin'], X['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    X['doy_sin'], X['doy_cos'] = np.sin(2*np.pi*(doy-1)/365.25), np.cos(2*np.pi*(doy-1)/365.25)
    lat, lon = d.lat.to_numpy(), d.lon.to_numpy()
    elev, az, cosz = solar_position(mid, lat, lon)
    X = X.copy()  # Consolidate before adding the final solar columns.
    X['solar_elevation'], X['solar_cos_zenith'] = elev, cosz
    X['solar_azimuth_sin'], X['solar_azimuth_cos'] = np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))
    # Retain sunrise/sunset intervals even when their midpoint is below horizon.
    start = solar_position(pd.DatetimeIndex(d.ts)-pd.Timedelta(hours=1), lat, lon)[0]
    end = solar_position(pd.DatetimeIndex(d.ts), lat, lon)[0]
    sunlit = np.maximum.reduce([elev, start, end]) > 0
    for c in X:
        X[c] = X[c].astype('int16' if c in categorical else 'float32')
    if any(np.isinf(X[c].to_numpy()).any() for c in X):
        raise AssertionError('Infinite feature')
    return X, categorical, sunlit, d.capacity_kw.to_numpy(dtype='float32')
