"""Reviewed unit transforms from SJ; fit statistics are implemented separately."""
import numpy as np

def transform_rule(name):
    if name.endswith('__age_days'):
        return 'log1p', 1.
    if name == 'capacity_kw':
        return 'log1p', 2.
    if name in ['wa_dist', 'wb_dist']:
        return 'log1p', 2.
    if name in ['lat', 'lon']:
        return 'identity', 5.
    if name.endswith(('__missing17', '__both_original', '__out_of_range')):
        return 'identity', 1.
    # Current values, differences and every daily descriptor share physical scale.
    if any(token in name for token in ['cloud', 'humidity', 'rel_hum', '_prob']):
        return 'identity', 100.
    if any(token in name for token in ['rain', 'snow', 'ice', 'total_liq']):
        return 'signed_log1p', 1.
    if any(token in name for token in ['uv_idx']):
        return 'identity', 10.
    if name in ['wb_ceiling', 'wb_vis']:
        return 'log1p', 3.
    if 'press' in name:
        return 'identity', 100.
    if any(token in name for token in ['wind_speed', 'wind_gust', 'gust_minus']):
        return 'identity', 10.
    if name == 'ab_wind_dir_difference':
        return 'identity', 180.
    if any(token in name for token in ['temp', 'dew_point']):
        return 'identity', 30.
    if name == 'hour':
        return 'identity', 24.
    if name == 'solar_elevation':
        return 'identity', 90.
    if any(token in name for token in ['_sin', '_cos']):
        return 'identity', 1.
    return 'identity', None

def feature_transform(values, name):
    rule, _ = transform_rule(name)
    if rule == 'log1p':
        return np.log1p(np.maximum(values, 0))
    if rule == 'signed_log1p':
        return np.sign(values)*np.log1p(np.abs(values))
    return values
