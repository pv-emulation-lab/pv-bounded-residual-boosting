"""
Physical reference simulation retained from the supplied manuscript package.

Input : data/nasa-power-hourly.csv  (hourly NASA POWER data, UTC, as downloaded)
Output: daily reference energy E (kWh/day per 1 kWp) and daily weather predictors,
        indexed by the local Philippine date (Asia/Manila, UTC+08:00).
Reproduced with Python 3.12.14 and pvlib 0.16.1.
"""
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.modelchain import ModelChain

# ---------------------------------------------------------------- site
LAT, LON, ALT = 8.486, 124.657, 113.37          # USTP Cagayan de Oro; NASA POWER grid elevation (m)
location = Location(LAT, LON, tz="Asia/Manila", altitude=ALT)

# ------------------------------------------- reference system (Table 3)
BASE = dict(
    pdc0=1000,           # W, DC nameplate at STC (1 kWp)
    gamma=-0.0035,       # 1/degC, power temperature coefficient
    tilt=10,             # degrees
    az=180,              # degrees, south-facing
    albedo=0.2,
    dcac=1.1,            # DC/AC ratio
    eta=0.96,            # nominal inverter efficiency
    loss=None,           # None = pvlib PVWatts default losses (about 14.08 %)
    irr="as_downloaded", # or "closure_dni" (sensitivity variant)
)


def load_weather(path="data/nasa-power-hourly.csv"):
    raw = pd.read_csv(path, parse_dates=["time_utc"], index_col="time_utc")
    raw.index = raw.index.tz_localize("UTC")
    return raw


def simulate(cfg=BASE, raw=None):
    """Return a daily DataFrame: reference energy E and predictors, one row per complete local day."""
    raw = load_weather() if raw is None else raw

    # NASA hourly values are hour averages stamped at the start of the hour (UTC):
    # evaluate the solar geometry at the middle of each interval (+30 min).
    w = raw.rename(columns={"ALLSKY_SFC_SW_DWN": "ghi", "ALLSKY_SFC_SW_DNI": "dni",
                            "ALLSKY_SFC_SW_DIFF": "dhi", "T2M": "temp_air",
                            "WS10M": "wind_speed"}).copy()
    w.index = w.index + pd.Timedelta(minutes=30)

    if cfg["irr"] == "closure_dni":      # sensitivity variant: make GHI = DHI + DNI*cos(zenith) close
        cosz = np.cos(np.radians(location.get_solarposition(w.index).zenith)).clip(lower=0)
        w["dni"] = np.where(cosz > 0.1, ((w.ghi - w.dhi) / cosz.replace(0, np.nan)).clip(0, 1400), 0.0)

    array = Array(
        FixedMount(cfg["tilt"], cfg["az"]),
        albedo=cfg["albedo"],
        module_parameters=dict(pdc0=cfg["pdc0"], gamma_pdc=cfg["gamma"]),
        temperature_model_parameters=pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"],
    )
    system = PVSystem(
        arrays=[array],
        inverter_parameters=dict(pdc0=cfg["pdc0"] / cfg["dcac"], eta_inv_nom=cfg["eta"]),
        losses_parameters=cfg["loss"],
    )
    mc = ModelChain.with_pvwatts(system, location, aoi_model="physical",
                                 spectral_model="no_loss", temperature_model="sapm")
    mc.run_model(w[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])

    p_ac = mc.results.ac.clip(lower=0).values                 # W, negative output set to zero
    poa = mc.results.total_irrad["poa_global"].values         # W/m2, plane of array

    # Local Philippine clock (UTC+08:00, no daylight saving) and daily aggregation
    local = (raw.index + pd.Timedelta(hours=8)).tz_localize(None)
    hourly = pd.DataFrame({"p": p_ac, "poa": poa,
                           "ghi": raw.ALLSKY_SFC_SW_DWN.values, "t": raw.T2M.values,
                           "rh": raw.RH2M.values, "ws": raw.WS10M.values,
                           "pr": raw.PRECTOTCORR.values}, index=local)
    daily = hourly.groupby(hourly.index.normalize()).agg(
        E=("p", lambda x: x.sum() / 1000),        # Wh -> kWh (hourly steps of 1 h), Eq. (3)
        H_poa=("poa", lambda x: x.sum() / 1000),  # kWh/m2/day
        GHI_d=("ghi", lambda x: x.sum() / 1000),  # kWh/m2/day
        T_mean=("t", "mean"), RH_mean=("rh", "mean"), WS_mean=("ws", "mean"),
        P_total=("pr", "sum"), n=("p", "size"))
    daily = daily[daily.n == 24].drop(columns="n")             # keep complete local days only

    doy = daily.index.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return daily


if __name__ == "__main__":
    d = simulate()
    d.to_csv("results/daily-reference.csv")
    print(d.E.describe().round(2))
