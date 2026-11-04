"""Reviewed original 01_state NOAA geometry; original source untouched."""
import numpy as np
import pandas as pd

def solar(ts: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray):
    """태양 고도·방위 (NOAA 근사, 도 단위)."""
    doy = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    # KST(UTC+9) 기준 진태양시. ts는 우측닫힘이므로 구간 중앙(-30분)을 쓴다.
    tst = (hour - 0.5) * 60 + eqtime + 4 * (lon - 135.0)
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    elev = 90.0 - np.degrees(np.arccos(cosz))
    az = np.degrees(np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))) % 360
    return elev.astype("float32"), az.astype("float32"), cosz.astype("float32")
