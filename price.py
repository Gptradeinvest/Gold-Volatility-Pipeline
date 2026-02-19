"""
GOLD VOLATILITY PIPELINE — GLD · XAU/USD
yfinance + matplotlib  ·  D1
==========================================
Usage:
    pip install yfinance scipy numpy pandas matplotlib
    python price.py

Output:
    gold_surface_data.json   — données brutes
    gold_report.pdf          — rapport 4 pages (+ fenêtre bayésienne)
    gold_report.png          — aperçu page 4 (Bayes)
"""

import json
import logging
import math
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

try:
    import pandas as pd
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Ciblé: uniquement les catégories bruyantes connues des dépendances
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Matplotlib.*")
warnings.filterwarnings("ignore", message=".*perf_counter.*")

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Erreur fatale pipeline — remplace SystemExit pour intégration."""
    pass


# Fetch yfinance avec timeout threading
_FETCH_TIMEOUT = 30  # secondes


def _yf_fetch_timeout(fn, *args, timeout=_FETCH_TIMEOUT, **kwargs):
    """Exécute fn(*args, **kwargs) dans un thread avec timeout."""
    result = [None]
    exc    = [None]

    def _run():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise PipelineError(f"Timeout yfinance ({timeout}s): {fn.__name__}")
    if exc[0] is not None:
        raise exc[0]
    return result[0]

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
TICKER_GLD    = "GLD"
TICKER_GC     = "GC=F"       # spot XAU proxy uniquement (pas d'options)
TICKER_GVZ    = "^GVZ"
TICKER_RATE   = "^IRX"
HIST_YEARS    = 6
MIN_OI_GLD    = 10
MAX_SPREAD_GLD = 0.30
MAX_IV        = 3.0
MIN_IV        = 0.005
DEFAULT_JSON  = "gold_surface_data.json"
DEFAULT_PDF   = "gold_report.pdf"

# numpy trapz compat (np.trapz → np.trapezoid en numpy 2.x)
try:
    _np_trapz = np.trapezoid
except AttributeError:
    _np_trapz = np.trapz

# ─────────────────────────────────────────────────────────────────
#  MATH BS
# ─────────────────────────────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_price(S, K, T, sig, r, is_call):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if is_call:
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_vega(S, K, T, sig, r=0.0):
    if T <= 0 or sig <= 0:
        return 1e-10
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return S * math.sqrt(T) * norm_pdf(d1)

def implied_vol(price, S, K, T, r, is_call, lo=1e-4, hi=5.0, tol=1e-7, maxiter=120):
    if T <= 0 or price <= 0:
        return None
    # Intrinsèque correct: valeur actualisée (forward intrinsic)
    fwd_intrinsic = max(0.0, (S - K * math.exp(-r * T)) if is_call else (K * math.exp(-r * T) - S))
    if price < fwd_intrinsic - 0.01:
        return None
    p_lo = bs_price(S, K, T, lo, r, is_call)
    p_hi = bs_price(S, K, T, hi, r, is_call)
    if not (p_lo <= price <= p_hi):
        return None
    for _ in range(maxiter):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, mid, r, is_call)
        if abs(p - price) < tol:
            break
        if p > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    iv = (lo + hi) / 2
    # Options à prix négligeable (deep OTM + low vol): inversion numériquement dégénérée
    if price < tol:
        return None
    for _ in range(10):
        p = bs_price(S, K, T, iv, r, is_call)
        v = bs_vega(S, K, T, iv, r)
        if v < 1e-10:
            break
        delta_iv = (p - price) / v
        iv = max(lo, min(hi, iv - delta_iv))
        if abs(delta_iv) < tol:
            break
    return iv if MIN_IV <= iv <= MAX_IV else None

def bs_delta(S, K, T, sig, r, is_call):
    if T <= 0 or sig <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0

def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────
#  FETCH — SPOTS + TAUX
# ─────────────────────────────────────────────────────────────────
def _ticker_history(ticker_str, period):
    return yf.Ticker(ticker_str).history(period=period)

def fetch_spots_rate():
    print("→ Spots GLD / GC / taux...")
    try:
        h_gld = _yf_fetch_timeout(_ticker_history, TICKER_GLD, "5d")
        S_gld = float(h_gld["Close"].iloc[-1])
        if S_gld <= 0:
            raise PipelineError("GLD spot ≤ 0 — données corrompues")
    except Exception as e:
        raise PipelineError(f"GLD spot indisponible: {e}") from e
    try:
        gc_h  = _yf_fetch_timeout(_ticker_history, TICKER_GC, "5d")
        S_gc  = float(gc_h["Close"].iloc[-1]) if len(gc_h) > 0 else None
    except Exception as e:
        log.warning("GC=F indisponible: %s", e)
        S_gc = None
    try:
        irx_h = _yf_fetch_timeout(_ticker_history, TICKER_RATE, "5d")
        r     = float(irx_h["Close"].iloc[-1]) / 100.0 if len(irx_h) > 0 else 0.05
    except Exception as e:
        log.warning("^IRX indisponible, fallback r=5%%: %s", e)
        r = 0.05
    ratio  = S_gc / S_gld if S_gc else 10.28
    gc_str = f"${S_gc:.2f}" if S_gc is not None else "indisponible"
    print(f"   GLD = ${S_gld:.2f}  GC = {gc_str}  ratio = {ratio:.4f}  r = {r*100:.2f}%")
    return {"S_gld": S_gld, "S_gc": S_gc, "ratio": ratio, "r": r}

def fetch_gvz():
    print("→ GVZ...")
    try:
        h = _yf_fetch_timeout(_ticker_history, TICKER_GVZ, "5d")
        v = float(h["Close"].iloc[-1]) if len(h) > 0 else None
        print(f"   GVZ = {v:.2f}%" if v else "   GVZ indisponible")
        return v
    except Exception as e:
        log.warning("GVZ erreur: %s", e)
        return None

# ─────────────────────────────────────────────────────────────────
#  FETCH — OPTIONS CHAIN
# ─────────────────────────────────────────────────────────────────
def _process_df(df, S, T, r, is_call, min_oi, max_spread):
    rows = []
    for _, row in df.iterrows():
        K = safe_float(row.get("strike"), 0.0)
        if K <= 0 or K < S * 0.55 or K > S * 1.65:
            continue
        oi = int(safe_float(row.get("openInterest", 0), 0.0))
        if oi < min_oi:
            continue
        bid = safe_float(row.get("bid", 0), 0.0)
        ask = safe_float(row.get("ask", 0), 0.0)
        if ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        if mid <= 0:
            continue
        spread_rel = (ask - bid) / mid
        if spread_rel > max_spread:
            continue
        iv = implied_vol(mid, S, K, T, r, is_call)
        if iv is None:
            continue
        delta = bs_delta(S, K, T, iv, r, is_call)
        rows.append({"K": K, "iv": round(iv*100, 4), "delta": round(delta, 5),
                     "mid": round(mid, 4), "spread_rel": round(spread_rel, 4),
                     "oi": oi, "type": "call" if is_call else "put"})
    return rows

def _interp_atm(rows, S, T, r):
    if not rows:
        return None
    F = S * math.exp(r * T)
    s = sorted(rows, key=lambda x: x["K"])
    lo = [x for x in s if x["K"] <= F]
    hi = [x for x in s if x["K"] > F]
    if not lo or not hi:
        return min(s, key=lambda x: abs(x["K"] - F))["iv"] / 100.0
    l, u = lo[-1], hi[0]
    if u["K"] == l["K"]:
        return l["iv"] / 100.0
    w = math.log(F / l["K"]) / math.log(u["K"] / l["K"])
    return ((1 - w) * l["iv"] + w * u["iv"]) / 100.0

def _interp_delta(rows, target):
    if not rows:
        return None
    lo_r = [x for x in rows if x["delta"] <= target] if target > 0 else [x for x in rows if x["delta"] >= target]
    hi_r = [x for x in rows if x["delta"] > target]  if target > 0 else [x for x in rows if x["delta"] < target]
    if not lo_r or not hi_r:
        return min(rows, key=lambda x: abs(x["delta"] - target))["iv"] / 100.0
    l = min(lo_r, key=lambda x: abs(x["delta"] - target))
    u = min(hi_r, key=lambda x: abs(x["delta"] - target))
    d = abs(u["delta"] - l["delta"])
    if d < 1e-6:
        return (l["iv"] + u["iv"]) / 200.0
    w = abs(target - l["delta"]) / d
    return ((1 - w) * l["iv"] + w * u["iv"]) / 100.0

def fetch_surface(ticker_str, S, r, label, min_oi, max_spread, max_t=400):
    print(f"→ Options chain {label} ({ticker_str})...")
    try:
        tkr         = yf.Ticker(ticker_str)
        expirations = _yf_fetch_timeout(lambda: tkr.options)
    except Exception as e:
        log.warning("%s options indisponible: %s", label, e)
        return {"tenors": [], "surface": []}
    if not expirations:
        print(f"   {label}: aucune expiration")
        return {"tenors": [], "surface": []}
    today = datetime.now(timezone.utc).date()
    tenors, raw = [], []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        T_cal = (exp_date - today).days
        if T_cal <= 1:
            continue
        if T_cal > max_t:
            break
        T = T_cal / 365.0
        try:
            chain = _yf_fetch_timeout(tkr.option_chain, exp_str)
        except Exception as e:
            log.warning("option_chain %s %s: %s", label, exp_str, e)
            continue
        c_rows = _process_df(chain.calls, S, T, r, True,  min_oi, max_spread)
        p_rows = _process_df(chain.puts,  S, T, r, False, min_oi, max_spread)
        all_r  = c_rows + p_rows
        if len(all_r) < 3:
            continue
        for rr in all_r:
            raw.append({"tenor_days": T_cal, **rr})
        atm  = _interp_atm(all_r, S, T, r)
        i25c = _interp_delta(c_rows, 0.25)
        i25p = _interp_delta(p_rows, -0.25)
        rr25 = (i25c - i25p) * 100 if (i25c is not None and i25p is not None and atm is not None) else None
        bf25 = ((i25c + i25p) / 2 - atm) * 100 if (i25c is not None and i25p is not None and atm is not None) else None
        tenors.append({
            "expiry": exp_str, "tenor_days": T_cal, "T": round(T, 6),
            "atm_iv":  round(atm * 100, 4) if atm is not None else None,
            "iv_25c":  round(i25c * 100, 4) if i25c is not None else None,
            "iv_25p":  round(i25p * 100, 4) if i25p is not None else None,
            "rr25":    round(rr25, 4) if rr25 is not None else None,
            "bf25":    round(bf25, 4) if bf25 is not None else None,
            "n_strikes": len(all_r), "n_calls": len(c_rows), "n_puts": len(p_rows),
        })
        atm_s = f"{atm*100:.2f}%" if atm else "?"
        rr_s  = f"{rr25:.2f}%" if rr25 is not None else "?"
        bf_s  = f"{bf25:.2f}%" if bf25 is not None else "?"
        print(f"   {exp_str}: {T_cal:3d}d | ATM={atm_s} | RR25={rr_s} | BF25={bf_s} | n={len(all_r)}")
    tenors.sort(key=lambda x: x["tenor_days"])
    print(f"   → {label}: {len(tenors)} tenors, {len(raw)} quotes")
    return {"tenors": tenors, "surface": raw}

# ─────────────────────────────────────────────────────────────────
#  FETCH — RV CÔNE
# ─────────────────────────────────────────────────────────────────
def fetch_rv(ticker_str, label):
    print(f"→ RV cône {label}...")
    try:
        hist = _yf_fetch_timeout(_ticker_history, ticker_str, f"{HIST_YEARS}y")
    except Exception as e:
        log.warning("%s historique erreur: %s", label, e)
        return {"cone": {}, "rv_recent": {}, "jumps": [], "stats": {}}
    closes = hist["Close"].dropna()
    if len(closes) < 30:
        return {"cone": {}, "rv_recent": {}, "jumps": [], "stats": {}}
    log_ret = np.log(closes / closes.shift(1)).dropna()
    windows = [7, 14, 21, 30, 60, 90, 120, 180]
    cone = {}
    for w in windows:
        rv = log_ret.rolling(w).std().dropna() * math.sqrt(252) * 100
        if len(rv) < 20:
            continue
        a = rv.values
        cone[str(w)] = {
            "p10": round(float(np.percentile(a, 10)), 2),
            "p25": round(float(np.percentile(a, 25)), 2),
            "p50": round(float(np.percentile(a, 50)), 2),
            "p75": round(float(np.percentile(a, 75)), 2),
            "p90": round(float(np.percentile(a, 90)), 2),
            "current": round(float(a[-1]), 2), "n_obs": len(a),
        }
        print(f"   {label} RV {w:3d}d: p50={cone[str(w)]['p50']:.1f}% current={cone[str(w)]['current']:.1f}%")
    rv_recent = {}
    for w in windows:
        if len(log_ret) >= w:
            rv_recent[str(w)] = round(float(log_ret.iloc[-w:].std() * math.sqrt(252) * 100), 2)
    std60 = log_ret.rolling(60).std().replace(0, np.nan)
    z = (log_ret / std60).dropna()
    idx = log_ret.index.intersection(z.index)
    lr  = log_ret.loc[idx]; z2 = z.loc[idx]
    mask = abs(z2) > 3.0; jd = lr[mask]
    jumps = [{"date": str(dt.date()) if hasattr(dt, "date") else str(dt)[:10],
               "return": round(float(ret) * 100, 2), "zscore": round(float(z2[dt]), 2)}
              for dt, ret in jd.items()]
    jumps.sort(key=lambda x: x["date"], reverse=True)
    jumps = jumps[:20]
    ra = log_ret.values
    stats = {
        "mean_annual": round(float(np.mean(ra)) * 252 * 100, 2),
        "rv_annual":   round(float(np.std(ra)) * math.sqrt(252) * 100, 2),
        "skew":        round(float(pd.Series(ra).skew()), 4),
        "kurt_excess": round(float(pd.Series(ra).kurtosis()), 4),
        "n_days":      len(ra),
        "start_date":  str(closes.index[0].date()),
        "end_date":    str(closes.index[-1].date()),
    }
    print(f"   {label}: {len(jumps)} sauts 3σ")
    return {"cone": cone, "rv_recent": rv_recent, "jumps": jumps, "stats": stats}

# ─────────────────────────────────────────────────────────────────
#  COMPUTE — IVR/IVP + FORWARD VOL + SMILES
# ─────────────────────────────────────────────────────────────────
def compute_ivr_ivp(tenors, rv_cone):
    cone_w  = [int(k) for k in rv_cone.keys()]
    rv_curr = {int(k): v.get("current") for k, v in rv_cone.items()}
    results = []
    for t in tenors:
        iv = t.get("atm_iv")
        if iv is None:
            continue
        td  = t["tenor_days"]
        cw  = min(cone_w, key=lambda w: abs(w - td)) if cone_w else None
        rv_c = rv_curr.get(cw) if cw else None
        ivr  = round(iv / rv_c, 4) if rv_c and rv_c > 0 else None
        ivp  = None
        if cw and str(cw) in rv_cone:
            c   = rv_cone[str(cw)]
            pts = [(10, c["p10"]), (25, c["p25"]), (50, c["p50"]), (75, c["p75"]), (90, c["p90"])]
            if   iv <= pts[0][1]:  ivp = 10.0
            elif iv >= pts[-1][1]: ivp = 90.0
            else:
                for i in range(len(pts) - 1):
                    p0, v0 = pts[i]; p1, v1 = pts[i+1]
                    if v0 <= iv <= v1:
                        dv = v1 - v0
                        ivp = p0 + (iv - v0) / dv * (p1 - p0) if dv > 1e-10 else (p0 + p1) / 2
                        break
        results.append({"tenor_days": td, "iv": round(iv, 3), "rv_current": rv_c, "ivr": ivr,
            "ivp": round(ivp, 1) if ivp is not None else None,
            "signal": "SELL VOL" if (ivr is not None and ivr > 1.15) else
                      "BUY VOL"  if (ivr is not None and ivr < 0.85) else "NEUTRAL"})
    return results

def compute_fwd_vols(tenors):
    v = [t for t in tenors if t.get("atm_iv")]
    res = []
    for i in range(1, len(v)):
        t1, t2 = v[i-1], v[i]
        T1, T2 = t1["T"], t2["T"]
        s1, s2 = t1["atm_iv"] / 100, t2["atm_iv"] / 100
        fv2 = (T2 * s2**2 - T1 * s1**2) / (T2 - T1) if T2 > T1 else None
        fwd_vol_val = round(math.sqrt(fv2) * 100, 3) if (fv2 is not None and fv2 > 0) else None
        res.append({"from_days": t1["tenor_days"], "to_days": t2["tenor_days"],
            "fwd_vol": fwd_vol_val,
            "cal_arb_ok": bool(fv2 is not None and fv2 > 0)})
    return res

def build_smiles(raw, S, max_t=8):
    by_t = {}
    for r in raw:
        by_t.setdefault(r["tenor_days"], []).append(r)
    result = []
    for td in sorted(by_t.keys()):
        rows = sorted(by_t[td], key=lambda x: x["K"])
        result.append({"tenor_days": td, "smile": [
            {"k": round(math.log(r["K"] / S) * 100, 2),
             "iv": r["iv"], "K": r["K"], "delta": r["delta"], "type": r["type"]}
            for r in rows]})
        if len(result) >= max_t:
            break
    return result

# ─────────────────────────────────────────────────────────────────
#  QA AUDIT
# ─────────────────────────────────────────────────────────────────
def audit_qa(tenors, rv_cone, label):
    n_a = sum(1 for t in tenors if t.get("atm_iv"))
    n_s = sum(1 for t in tenors if t.get("rr25") is not None)
    n_k = sum(t.get("n_strikes", 0) for t in tenors)
    score = 100; warns = []
    if n_a < 4:
        warns.append({"severity": "CRITIQUE", "msg": f"{n_a} tenors ATM"}); score -= 30
    elif n_a < 6:
        warns.append({"severity": "MAJEUR", "msg": f"{n_a} tenors ATM"}); score -= 10
    if n_s < 3:
        warns.append({"severity": "MAJEUR", "msg": f"{n_s} tenors RR25"}); score -= 15
    vl = [(t["tenor_days"], t["atm_iv"]) for t in tenors if t.get("atm_iv")]
    for i in range(1, len(vl)):
        t1, iv1 = vl[i-1]; t2, iv2 = vl[i]
        if t2 / 365 * iv2**2 < t1 / 365 * iv1**2:
            warns.append({"severity": "MAJEUR", "msg": f"Cal arb {t1}→{t2}d"}); score -= 8
    if len(rv_cone) < 4:
        warns.append({"severity": "MODÉRÉ", "msg": "RV cône < 4 fenêtres"}); score -= 5
    return {"label": label, "score": max(0, score), "n_tenors_atm": n_a,
            "n_tenors_smile": n_s, "n_strikes": n_k, "warnings": warns}

# ─────────────────────────────────────────────────────────────────
#  FETCH PIPELINE — construit le dict data complet
# ─────────────────────────────────────────────────────────────────
def run_fetch(json_out=DEFAULT_JSON):
    if not HAS_YFINANCE:
        print("ERREUR: yfinance non installé. pip install yfinance pandas")
        raise PipelineError("yfinance non installé. pip install yfinance pandas")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n{'='*62}")
    print(f"GOLD MULTI-INSTRUMENT FETCHER — {ts}")
    print(f"{'='*62}\n")

    spots = fetch_spots_rate()
    S_gld, S_gc, ratio, r = spots["S_gld"], spots["S_gc"], spots["ratio"], spots["r"]
    gvz = fetch_gvz()

    # ── GLD — surface options complète ───────────────────────────
    surf_gld   = fetch_surface(TICKER_GLD, S_gld, r, "GLD", MIN_OI_GLD, MAX_SPREAD_GLD)
    rv_gld     = fetch_rv(TICKER_GLD, "GLD")
    ivr_gld    = compute_ivr_ivp(surf_gld["tenors"], rv_gld["cone"])
    fwd_gld    = compute_fwd_vols(surf_gld["tenors"])
    smiles_gld = build_smiles(surf_gld["surface"], S_gld)
    qa_gld     = audit_qa(surf_gld["tenors"], rv_gld["cone"], "GLD")

    # ── XAU/USD — surface GLD rescalée, RV = GLD (corrélation ~1) ─
    # GC=F utilisé seulement pour le prix spot XAU (basis ~0)
    # IV% invariante par scaling → surface XAU = surface GLD
    ivr_xau = compute_ivr_ivp(surf_gld["tenors"], rv_gld["cone"])
    fwd_xau = compute_fwd_vols(surf_gld["tenors"])
    qa_xau  = audit_qa(surf_gld["tenors"], rv_gld["cone"], "XAU/USD (GLD rescalé)")

    data = {
        "meta": {
            "fetched_at":    ts,
            "source":        "yfinance · GLD options · GC=F spot · ^GVZ · ^IRX",
            "gvz":           round(gvz, 2) if gvz else None,
            "rate":          round(r, 5),
            "ratio_gld_xau": round(ratio, 4),
            "note": "GLD=options ETF liquides. XAU spot=GC=F proxy. IV%=GLD (invariante par scaling).",
        },
        "gld": {
            "spot":      round(S_gld, 2),
            "surface":   surf_gld["tenors"],
            "smiles":    smiles_gld,
            "rv_cone":   rv_gld["cone"],
            "rv_recent": rv_gld["rv_recent"],
            "rv_stats":  rv_gld["stats"],
            "jumps":     rv_gld["jumps"],
            "ivr_ivp":   ivr_gld,
            "fwd_vols":  fwd_gld,
            "quality":   qa_gld,
        },
        "xauusd": {
            "spot":        round(S_gc, 2) if S_gc else round(S_gld * ratio, 2),
            "spot_source": "GC=F front-month (proxy XAU spot, basis ~0)",
            "ratio_gld":   round(ratio, 4),
            "surface":     surf_gld["tenors"],
            "smiles":      smiles_gld,
            "rv_cone":     rv_gld["cone"],
            "rv_recent":   rv_gld["rv_recent"],
            "rv_stats":    rv_gld["stats"],
            "jumps":       rv_gld["jumps"],
            "ivr_ivp":     ivr_xau,
            "fwd_vols":    fwd_xau,
            "quality":     qa_xau,
            "note":        "Surface IV = GLD (IV% invariante). RV = GLD (corrélation ~1). Spot via GC=F.",
        },
        # vue consolidée GLD
        "spot":      round(S_gld, 2),
        "surface":   surf_gld["tenors"],
        "smiles":    smiles_gld,
        "rv_cone":   rv_gld["cone"],
        "rv_recent": rv_gld["rv_recent"],
        "rv_stats":  rv_gld["stats"],
        "jumps":     rv_gld["jumps"],
        "ivr_ivp":   ivr_gld,
        "fwd_vols":  fwd_gld,
        "quality":   qa_gld,
    }

    try:
        with open(json_out, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except OSError as e:
        raise PipelineError(f"Impossible d'écrire {json_out}: {e}") from e

    print(f"\n{'='*62}")
    print(f"JSON → {json_out}")
    xau_s = f"${S_gc:.2f}" if S_gc is not None else f"${S_gld*ratio:.2f} (estimé)"
    gvz_s = f"{gvz:.2f}%" if gvz is not None else "N/A"
    print(f"  GLD ${S_gld:.2f} | XAU/USD {xau_s} | ratio {ratio:.4f}")
    print(f"  GVZ {gvz_s} | r {r*100:.2f}%")
    for lbl, qa in [("GLD", qa_gld), ("XAU", qa_xau)]:
        print(f"  [{lbl}] ATM:{qa['n_tenors_atm']} Strikes:{qa['n_strikes']} QA:{qa['score']}/100")
        for w in qa["warnings"]:
            print(f"       [{w['severity']:8s}] {w['msg']}")
    print(f"{'='*62}\n")
    return data

# ─────────────────────────────────────────────────────────────────
#  PALETTE + HELPERS GRAPHIQUES
# ─────────────────────────────────────────────────────────────────
BG      = "#080a0c"
PANEL   = "#0c0e13"
PANEL2  = "#0f1118"
BORDER  = "#1e2235"
GOLD    = "#c8a020"
GOLDL   = "#e6bc42"
GOLDD   = "#6a510a"
RED     = "#cc4444"
GREEN   = "#3a9660"
BLUE    = "#3d72b0"
ORANGE  = "#c07830"
VIOLET  = "#7b5ea7"
TEXT    = "#b2b8c6"
TEXTDIM = "#4e5567"
TEXTMID = "#8892a4"

PALETTE_TENORS = [GOLD, BLUE, GREEN, RED, VIOLET, ORANGE, GOLDL, TEXTMID]

plt.rcParams.update({
    "figure.facecolor":  BG, "axes.facecolor":   PANEL, "axes.edgecolor":  BORDER,
    "axes.labelcolor":   TEXTMID, "axes.titlecolor": TEXT,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": BORDER, "grid.linewidth": 0.5,
    "xtick.color": TEXTDIM, "ytick.color": TEXTDIM,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": TEXT, "font.family": "monospace", "font.size": 8,
    "legend.facecolor": PANEL2, "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT, "legend.fontsize": 7,
    "lines.linewidth": 1.8, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.facecolor": BG, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

def _panel_title(ax, title, sub=None):
    ax.text(0.0, 1.035, title, transform=ax.transAxes,
            color=TEXT, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    if sub:
        ax.text(1.0, 1.035, sub, transform=ax.transAxes,
                color=TEXTDIM, fontsize=6.5, va="bottom", ha="right")

def _badge(ax, text, color=GOLD, x=0.0, y=1.11):
    ax.text(x, y, f"  {text}  ", transform=ax.transAxes, color=color, fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color+"22",
                      edgecolor=color+"55", linewidth=0.5))

def _sparse_xticks(ax, labels, max_labels=12):
    n = len(labels)
    if n <= max_labels:
        return
    step = max(1, n // max_labels)
    sparse = [l if i % step == 0 else "" for i, l in enumerate(labels)]
    ax.set_xticklabels(sparse)

def _legend(ax, **kw):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(**kw)

# ─────────────────────────────────────────────────────────────────
#  RAPPORT — PAGE 1: Term Structure / RV Cône / IVR / Skew
# ─────────────────────────────────────────────────────────────────
def page1(data, instrument="gld"):
    instr    = data.get(instrument, data)
    meta     = data.get("meta", {})
    S        = instr.get("spot") or data.get("spot", 0)
    surface  = instr.get("surface") or data.get("surface", [])
    rv_cone  = instr.get("rv_cone") or data.get("rv_cone", {})
    ivr_ivp  = instr.get("ivr_ivp") or data.get("ivr_ivp", [])
    fwd_vols = instr.get("fwd_vols") or data.get("fwd_vols", [])
    gvz      = meta.get("gvz") or 0

    label_map   = {"gld": "GLD ETF", "gc": "GC=F Futures", "xauusd": "XAU/USD"}
    instr_label = label_map.get(instrument, instrument.upper())

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.text(0.03, 0.978, f"GOLD VOLATILITY REPORT — {instr_label}",
             color=GOLDL, fontsize=14, fontweight="bold", va="top", fontfamily="monospace")
    fig.text(0.03, 0.958,
             f"Spot: ${S:.2f}  |  Rate: {(meta.get('rate') or 0)*100:.2f}%  |  "
             f"GVZ: {gvz:.1f}%  |  {meta.get('fetched_at','')[:19].replace('T',' ')}  |  "
             f"Source: {meta.get('source','')}",
             color=TEXTDIM, fontsize=7.5, va="top", fontfamily="monospace")
    fig.text(0.97, 0.978, "PAGE 1 / 5",
             color=TEXTDIM, fontsize=7, va="top", ha="right", fontfamily="monospace")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           top=0.91, bottom=0.06, left=0.05, right=0.97,
                           hspace=0.55, wspace=0.32)

    # ── Term Structure ATM IV ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    valid = [(t["tenor_days"], t["atm_iv"]) for t in surface if t.get("atm_iv")]
    if valid:
        days, ivs = zip(*valid)
        ax1.plot(days, ivs, "o-", color=GOLD, lw=2.5, ms=4, zorder=5)
        ax1.fill_between(days, ivs, alpha=0.08, color=GOLD)
        for d, iv in zip(days, ivs):
            if d in (days[0], days[len(days)//2], days[-1]) or d <= 30:
                ax1.annotate(f"{iv:.1f}%", (d, iv),
                             textcoords="offset points", xytext=(0, 6),
                             fontsize=6, color=GOLDL, ha="center")
        if gvz:
            ax1.axhline(gvz, color=ORANGE, lw=1, ls="--", label=f"GVZ {gvz:.1f}%")
        iv_min = min(ivs); iv_max = max(ivs)
        margin = (iv_max - iv_min) * 0.15 + 1.0
        ax1.set_ylim(max(0, iv_min - margin), iv_max + margin)
        _legend(ax1, framealpha=0.3)
    ax1.set_xlabel("Tenor (jours)")
    ax1.set_ylabel("ATM IV (%)")
    _panel_title(ax1, "TERM STRUCTURE ATM IV", f"n={len(valid)} expirations")
    _badge(ax1, "IV RÉELLE", GOLD)

    # ── Risk Reversal 25Δ ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    rr_data = [(t["tenor_days"], t["rr25"]) for t in surface if t.get("rr25") is not None]
    if rr_data:
        d_rr, rr = zip(*rr_data)
        ax2.bar(range(len(d_rr)), rr,
                color=[GREEN if v >= 0 else RED for v in rr], alpha=0.85, width=0.6)
        ax2.set_xticks(range(len(d_rr)))
        ax2.set_xticklabels([f"{d}d" for d in d_rr], rotation=60, fontsize=5.5)
        _sparse_xticks(ax2, [f"{d}d" for d in d_rr], max_labels=10)
        ax2.axhline(0, color=BORDER, lw=1)
    _panel_title(ax2, "RISK REVERSAL 25Δ", "RR25 = IV(C25Δ)−IV(P25Δ)")
    _badge(ax2, "SKEW", BLUE)

    # ── RV Cône ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    cone_w = sorted([int(k) for k in rv_cone.keys()])
    if cone_w:
        p10  = [rv_cone[str(w)]["p10"]  for w in cone_w]
        p25  = [rv_cone[str(w)]["p25"]  for w in cone_w]
        p50  = [rv_cone[str(w)]["p50"]  for w in cone_w]
        p75  = [rv_cone[str(w)]["p75"]  for w in cone_w]
        p90  = [rv_cone[str(w)]["p90"]  for w in cone_w]
        curr = [rv_cone[str(w)]["current"] for w in cone_w]
        ax3.fill_between(cone_w, p10, p90, alpha=0.08, color=GOLD, label="p10-p90")
        ax3.fill_between(cone_w, p25, p75, alpha=0.12, color=GOLD, label="p25-p75")
        ax3.plot(cone_w, p50,  color=GOLD, lw=1.8, label="p50 historique")
        ax3.plot(cone_w, curr, color=BLUE, lw=2.2, marker="o", ms=4, label="RV actuelle")
        # IV ATM overlay
        iv_pts = []
        for w in cone_w:
            closest = min(valid, key=lambda x: abs(x[0]-w)) if valid else None
            iv_pts.append(closest[1] if closest else None)
        w_arr  = [cone_w[i] for i, v in enumerate(iv_pts) if v is not None]
        iv_arr = [v for v in iv_pts if v is not None]
        if w_arr:
            ax3.plot(w_arr, iv_arr, color=RED, lw=2, ls="--", marker="s", ms=3, label="IV ATM")
        _legend(ax3, loc="upper right", fontsize=6.5, ncol=2)
        ax3.set_xlabel("Fenêtre (jours)")
        ax3.set_ylabel("Vol annualisée (%)")
    _panel_title(ax3, "CÔNE RV HISTORIQUE vs IV ATM",
                 f"{instr.get('rv_stats',{}).get('start_date','')} → "
                 f"{instr.get('rv_stats',{}).get('end_date','')}")
    _badge(ax3, "RV RÉELLE", GREEN)
    _badge(ax3, "IV vs RV", RED, x=0.12)

    # ── IVR = IV / RV_current ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    if ivr_ivp:
        ivr_vals = [x["ivr"] for x in ivr_ivp if x.get("ivr") is not None]
        td_vals  = [x["tenor_days"] for x in ivr_ivp if x.get("ivr") is not None]
        if ivr_vals:
            ax4.bar(range(len(td_vals)), ivr_vals,
                    color=[RED if v > 1.15 else GREEN if v < 0.85 else GOLD for v in ivr_vals],
                    alpha=0.85, width=0.6)
            ax4.axhline(1.0,  color=TEXT,  lw=1,   ls="-")
            ax4.axhline(1.15, color=RED,   lw=0.8, ls="--", label="SELL +15%")
            ax4.axhline(0.85, color=GREEN, lw=0.8, ls="--", label="BUY −15%")
            ax4.set_xticks(range(len(td_vals)))
            ax4.set_xticklabels([f"{d}d" for d in td_vals], rotation=60, fontsize=5.5)
            _sparse_xticks(ax4, [f"{d}d" for d in td_vals], max_labels=10)
            _legend(ax4, fontsize=6)
    _panel_title(ax4, "IVR = IV_ATM / RV_current")
    _badge(ax4, "IV/RV RATIO", ORANGE)

    # ── Forward Vol implicite ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    if fwd_vols:
        labels_fv = [f"{f['from_days']}→{f['to_days']}" for f in fwd_vols if f.get("fwd_vol") is not None]
        vals_fv   = [f["fwd_vol"]   for f in fwd_vols if f.get("fwd_vol") is not None]
        ok_fv     = [f["cal_arb_ok"] for f in fwd_vols if f.get("fwd_vol") is not None]
        if labels_fv:
            ax5.bar(range(len(labels_fv)), vals_fv,
                    color=[GREEN if ok else RED for ok in ok_fv], alpha=0.8, width=0.6)
            if valid:
                ax5.axhline(valid[0][1], color=GOLD, lw=1, ls="--", alpha=0.6,
                            label=f"ATM 1er tenor {valid[0][1]:.1f}%")
            ax5.set_xticks(range(len(labels_fv)))
            ax5.set_xticklabels(labels_fv, rotation=55, fontsize=5.5)
            _sparse_xticks(ax5, labels_fv, max_labels=14)
            _legend(ax5, fontsize=6.5)
            for i, ok in enumerate(ok_fv):
                if not ok:
                    ax5.text(i, vals_fv[i] + 0.3, "ARB!", color=RED, fontsize=6, ha="center")
    _panel_title(ax5, "FORWARD VOL IMPLICITE", "σ_fwd(T1→T2) = √[(T2σ²−T1σ²)/(T2−T1)]")
    _badge(ax5, "VERT=arb-free  ROUGE=violation", GREEN)

    # ── Butterfly 25Δ convexité ───────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    bf_data = [(t["tenor_days"], t["bf25"]) for t in surface if t.get("bf25") is not None]
    if bf_data:
        d_bf, bf = zip(*bf_data)
        ax6.plot(d_bf, bf, "o-", color=VIOLET, lw=2, ms=3)
        ax6.fill_between(d_bf, bf, alpha=0.1, color=VIOLET)
        ax6.axhline(0, color=BORDER, lw=1)
    _panel_title(ax6, "BUTTERFLY 25Δ", "BF25 = (IV_25C+IV_25P)/2 − ATM")
    _badge(ax6, "CONVEXITÉ", VIOLET)

    fig.text(0.03, 0.012,
             "IV surface = quotes bid/ask mid réelles GLD options chain · "
             "RV = rolling std(log-returns)×√252 · IVR > 1.15 → SELL VOL · IVR < 0.85 → BUY VOL",
             color=TEXTDIM, fontsize=6.5, fontfamily="monospace")
    return fig

# ─────────────────────────────────────────────────────────────────
#  RAPPORT — PAGE 2: Smiles / Density / Greeks
# ─────────────────────────────────────────────────────────────────
def page2(data, instrument="gld"):
    instr   = data.get(instrument, data)
    meta    = data.get("meta", {})
    S       = instr.get("spot") or data.get("spot", 0)
    r       = float(meta.get("rate") or 0.05)
    smiles  = instr.get("smiles") or data.get("smiles", [])
    surface = instr.get("surface") or data.get("surface", [])
    label_map = {"gld": "GLD ETF", "gc": "GC=F Futures", "xauusd": "XAU/USD"}

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.text(0.03, 0.978, f"GOLD VOLATILITY REPORT — {label_map.get(instrument,'')} — SMILES & GREEKS",
             color=GOLDL, fontsize=13, fontweight="bold", va="top", fontfamily="monospace")
    fig.text(0.97, 0.978, "PAGE 2 / 5",
             color=TEXTDIM, fontsize=7, va="top", ha="right", fontfamily="monospace")

    gs = gridspec.GridSpec(3, 3, figure=fig, top=0.92, bottom=0.07,
                           left=0.05, right=0.97, hspace=0.58, wspace=0.34)

    # ── Smiles superposés ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    smile_tenors = smiles[:6]
    for i, sm in enumerate(smile_tenors):
        pts = sm.get("smile", [])
        if not pts:
            continue
        ax1.plot([p["k"] for p in pts], [p["iv"] for p in pts],
                 "o-", color=PALETTE_TENORS[i % len(PALETTE_TENORS)],
                 lw=1.8, ms=3, label=f"{sm['tenor_days']}d", alpha=0.9)
    ax1.axvline(0, color=TEXT, lw=0.7, ls="--", alpha=0.4)
    # Y cap au p95
    all_ivs = [p["iv"] for sm_t in smile_tenors for p in sm_t.get("smile", []) if p.get("iv")]
    if all_ivs:
        p95 = float(np.percentile(all_ivs, 95))
        ax1.set_ylim(bottom=max(0, min(all_ivs) * 0.90), top=min(p95 * 1.10, max(all_ivs)))
    ax1.set_xlabel("Log-moneyness k = ln(K/S) (%)")
    ax1.set_ylabel("IV (%)")
    _legend(ax1, fontsize=6.5, ncol=3, loc="upper right")
    _panel_title(ax1, "SMILES RÉELS — SUPERPOSITION", "Quotes bid/ask mid filtrées")
    _badge(ax1, "IV PAR STRIKE", GOLD)

    # ── Smile 30d Calls vs Puts ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    sm30 = next((s for s in smiles if abs(s["tenor_days"] - 30) < 15), None)
    if sm30:
        c_pts = [p for p in sm30["smile"] if p.get("type") == "call"]
        p_pts = [p for p in sm30["smile"] if p.get("type") == "put"]
        if c_pts:
            ax2.scatter([p["k"] for p in c_pts], [p["iv"] for p in c_pts],
                        color=BLUE, s=18, label="Calls", zorder=5, alpha=0.85)
        if p_pts:
            ax2.scatter([p["k"] for p in p_pts], [p["iv"] for p in p_pts],
                        color=RED, s=18, label="Puts", marker="^", zorder=5, alpha=0.85)
        ax2.axvline(0, color=TEXT, lw=0.7, ls="--", alpha=0.4)
        _legend(ax2, fontsize=6.5)
    _panel_title(ax2, f"SMILE {sm30['tenor_days'] if sm30 else 30}d CALLS vs PUTS")

    # ── Densité Risk-Neutral (Breeden-Litzenberger) ───────────────
    ax3 = fig.add_subplot(gs[1, :2])
    for i, tgt in enumerate([30, 60, 90]):
        sm = next((s for s in smiles if abs(s["tenor_days"] - tgt) < 20), None)
        if not sm or not sm.get("smile"):
            continue
        pts = sorted(sm["smile"], key=lambda x: x["K"])
        Ks  = np.array([p["K"] for p in pts])
        ivs = np.array([p["iv"] / 100 for p in pts])
        T   = sm["tenor_days"] / 365.0
        if len(Ks) < 5:
            continue
        K_grid  = np.linspace(Ks.min(), Ks.max(), max(len(Ks) * 2, 60))
        iv_grid = np.interp(K_grid, Ks, ivs)
        put_px  = np.array([bs_price(S, K, T, max(iv, 0.001), r, False)
                            for K, iv in zip(K_grid, iv_grid)])
        dK = K_grid[1] - K_grid[0]
        if dK < 1e-6:
            continue
        density = np.zeros(len(K_grid))
        density[1:-1] = np.maximum(0, np.diff(put_px, 2) / (dK**2) * math.exp(r * T))
        # Savitzky-Golay smoothing
        n_sg = len(density[1:-1])
        win  = min(21, n_sg if n_sg % 2 == 1 else n_sg - 1)
        if win >= 5:
            try:
                from scipy.signal import savgol_filter
                density[1:-1] = np.maximum(0, savgol_filter(density[1:-1], win, 3))
            except ImportError:
                kernel = np.ones(9) / 9
                density[1:-1] = np.maximum(0, np.convolve(density[1:-1], kernel, mode="same"))
        total = _np_trapz(density[1:-1], K_grid[1:-1])
        if total > 0:
            density /= total
        color = PALETTE_TENORS[i]
        ax3.fill_between(K_grid[1:-1], density[1:-1], alpha=0.15, color=color)
        ax3.plot(K_grid[1:-1], density[1:-1], color=color, lw=1.5, label=f"{sm['tenor_days']}d")
    ax3.axvline(S, color=GOLD, lw=1.2, ls="--", label=f"Spot ${S:.0f}")
    ax3.set_xlabel("Strike K ($)")
    ax3.set_ylabel("Densité q(K)")
    _legend(ax3, fontsize=6.5)
    _panel_title(ax3, "DENSITÉ RISK-NEUTRAL (Breeden-Litzenberger)", "q(K) ≈ e^{rT}·∂²C/∂K²")
    _badge(ax3, "DISTRIBUTION IMPLICITE", VIOLET)

    # ── Heatmap IV (Delta × Tenor) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    all_smile = smiles[:5]
    if all_smile and len(all_smile) >= 2:
        delta_grid = np.linspace(-0.5, 0.5, 20)
        tenor_grid = [sm["tenor_days"] for sm in all_smile]
        iv_mat = np.full((len(delta_grid), len(tenor_grid)), np.nan)
        for j, sm in enumerate(all_smile):
            for i_d, dg in enumerate(delta_grid):
                pts = sm.get("smile", [])
                if not pts:
                    continue
                closest = min(pts, key=lambda p: abs(p.get("delta", 0) - dg))
                if abs(closest.get("delta", 0) - dg) < 0.15:
                    iv_mat[i_d, j] = closest.get("iv", np.nan)
        masked = np.ma.masked_invalid(iv_mat)
        im = ax4.imshow(masked, aspect="auto", cmap="plasma",
                        extent=[0, len(tenor_grid)-1, delta_grid[-1], delta_grid[0]],
                        interpolation="bilinear")
        ax4.set_xticks(range(len(tenor_grid)))
        ax4.set_xticklabels([f"{t}d" for t in tenor_grid], fontsize=6)
        ax4.set_yticks([-0.4, -0.25, 0, 0.25, 0.4])
        ax4.set_ylabel("Delta")
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6, colors=TEXTDIM)
        cbar.set_label("IV (%)", fontsize=6.5, color=TEXTDIM)
    _panel_title(ax4, "SURFACE IV (Delta × Tenor)")
    _badge(ax4, "HEATMAP", ORANGE)

    # ── $ Gamma par strike — 30d ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    tenor30 = next((t for t in surface if abs(t["tenor_days"] - 30) < 20), None)
    if tenor30 and tenor30.get("atm_iv"):
        sig_atm = tenor30["atm_iv"] / 100
        T30     = tenor30["T"]
        Ks_g    = np.linspace(S * 0.70, S * 1.30, 80)
        dgamma  = []
        for K in Ks_g:
            if T30 <= 0 or sig_atm <= 0:
                dgamma.append(0); continue
            d1 = (math.log(S/K) + (r + 0.5*sig_atm**2)*T30) / (sig_atm*math.sqrt(T30))
            gamma = norm_pdf(d1) / (S * sig_atm * math.sqrt(T30))
            dgamma.append(0.5 * gamma * (S * 0.01)**2)
        ax5.fill_between(Ks_g, dgamma, alpha=0.2, color=GREEN)
        ax5.plot(Ks_g, dgamma, color=GREEN, lw=2)
        ax5.axvline(S, color=GOLD, lw=1, ls="--", label=f"Spot ${S:.0f}")
        ax5.set_xlabel("Strike K ($)")
        ax5.set_ylabel("$ Gamma ($/1%)")
        _legend(ax5, fontsize=7)
    _panel_title(ax5, "$ GAMMA PAR STRIKE — 30d", "½Γ(S×1%)²  [correction C1]")
    _badge(ax5, "DOLLAR GAMMA", GREEN)

    # ── Vega ATM par tenor ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    valid_surf = [(t["tenor_days"], t["atm_iv"]) for t in surface if t.get("atm_iv")]
    if valid_surf:
        ts_v, ivs_v = zip(*valid_surf)
        vega_atm = [S * math.sqrt(T_d/365) * norm_pdf(0) / 100 for T_d, _ in valid_surf]
        ax6.bar(range(len(ts_v)), vega_atm, color=BLUE, alpha=0.75, width=0.6)
        ax6.set_xticks(range(len(ts_v)))
        ax6.set_xticklabels([f"{d}d" for d in ts_v], rotation=55, fontsize=5.5)
        _sparse_xticks(ax6, [f"{d}d" for d in ts_v], max_labels=10)
        ax6.set_ylabel("Vega ATM ($/1%σ)")
    _panel_title(ax6, "VEGA ATM PAR TENOR", "S√T·N(0)/100")
    _badge(ax6, "VEGA", BLUE)

    fig.text(0.03, 0.012,
             "Smiles = quotes bid/ask mid filtrées (OI≥10, spread≤30%) · "
             "Density = Breeden-Litzenberger + Savitzky-Golay · $ Gamma = ½Γ(S×1%)²",
             color=TEXTDIM, fontsize=6.5, fontfamily="monospace")
    return fig

# ─────────────────────────────────────────────────────────────────
#  RAPPORT — PAGE 3: Comparaison / Jumps / Synthèse
# ─────────────────────────────────────────────────────────────────
def page3(data):
    meta  = data.get("meta", {})
    ratio = meta.get("ratio_gld_xau", 10.28)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.text(0.03, 0.978, "GOLD VOLATILITY REPORT — COMPARAISON INSTRUMENTS & STATISTIQUES",
             color=GOLDL, fontsize=13, fontweight="bold", va="top", fontfamily="monospace")
    fig.text(0.97, 0.978, "PAGE 3 / 5",
             color=TEXTDIM, fontsize=7, va="top", ha="right", fontfamily="monospace")

    gs = gridspec.GridSpec(3, 3, figure=fig, top=0.92, bottom=0.07,
                           left=0.05, right=0.97, hspace=0.58, wspace=0.34)

    # ── Comparaison ATM IV GLD vs XAU/USD ────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for key, color, label in [("gld", GOLD, "GLD ETF"), ("xauusd", RED, "XAU/USD (GLD rescalé)")]:
        surf = data.get(key, {}).get("surface", [])
        pts  = [(t["tenor_days"], t["atm_iv"]) for t in surf if t.get("atm_iv")]
        if pts:
            d, iv = zip(*pts)
            ax1.plot(d, iv, "o-", color=color, lw=2, ms=4, label=label, alpha=0.85)
    ax1.set_xlabel("Tenor (jours)")
    ax1.set_ylabel("ATM IV (%)")
    _legend(ax1, fontsize=7)
    _panel_title(ax1, "COMPARAISON ATM IV — GLD / XAU/USD",
                 "XAU surface = GLD rescalée (IV% invariante par scaling)")
    _badge(ax1, "GLD + XAU/USD", GOLD)

    # ── QA Scores ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    qa_data = [(lbl, data.get(k, {}).get("quality", {}))
               for k, lbl in [("gld","GLD"),("xauusd","XAU/USD")]]
    qa_data = [(lbl, q) for lbl, q in qa_data if q]
    if qa_data:
        labels_qa = [x[0] for x in qa_data]
        scores    = [x[1].get("score", 0) for x in qa_data]
        ax2.barh(range(len(labels_qa)), scores,
                 color=[GREEN if s>=80 else ORANGE if s>=60 else RED for s in scores],
                 alpha=0.85, height=0.5)
        ax2.set_yticks(range(len(labels_qa)))
        ax2.set_yticklabels(labels_qa, fontsize=9)
        ax2.set_xlim(0, 105)
        ax2.axvline(80, color=GREEN,  lw=0.8, ls="--")
        ax2.axvline(60, color=ORANGE, lw=0.8, ls="--")
        for i, (s, (_, q)) in enumerate(zip(scores, qa_data)):
            ax2.text(s+1, i, f"{s}/100  ({q.get('n_tenors_atm','—')} tenors)",
                     va="center", fontsize=7, color=TEXT)
    ax2.set_xlabel("Score QA")
    _panel_title(ax2, "SCORE QUALITÉ DONNÉES")
    _badge(ax2, "QA", GREEN)

    # ── Sauts GLD ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    jumps_gld = data.get("gld", {}).get("jumps", data.get("jumps", []))[:15]
    if jumps_gld:
        dates    = [j["date"]   for j in jumps_gld]
        rets     = [j["return"] for j in jumps_gld]
        zscores  = [j["zscore"] for j in jumps_gld]
        xs = range(len(dates))
        ax3.bar(xs, rets, color=[RED if ret_v < 0 else GREEN for ret_v in rets],
                alpha=0.85, width=0.6)
        ax3.axhline(0, color=BORDER, lw=1)
        ax3.set_xticks(xs)
        ax3.set_xticklabels(dates, rotation=55, fontsize=6)
        ax3.set_ylabel("Return (%)")
        for i, (ret, z) in enumerate(zip(rets, zscores)):
            ax3.text(i, ret + (0.3 if ret >= 0 else -0.5),
                     f"z={z:.1f}", ha="center", fontsize=5.5, color=TEXTDIM)
    _panel_title(ax3, f"SAUTS GLD — |z| > 3σ (rolling 60d) — {len(jumps_gld)} événements récents")
    _badge(ax3, "JUMP DETECTION", RED)

    # ── RV GLD (= XAU/USD, corrélation ~1) ───────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    cone = data.get("gld", {}).get("rv_cone", {})
    if cone:
        ws   = sorted([int(k) for k in cone.keys()])
        p50  = [cone[str(w)]["p50"]     for w in ws]
        p10  = [cone[str(w)]["p10"]     for w in ws]
        p90  = [cone[str(w)]["p90"]     for w in ws]
        curr = [cone[str(w)]["current"] for w in ws]
        ax4.fill_between(ws, p10, p90, alpha=0.08, color=GOLD, label="p10-p90")
        ax4.plot(ws, p50,  color=GOLD,  lw=1.5, ls="--", alpha=0.6, label="p50 historique")
        ax4.plot(ws, curr, color=GREEN, lw=2,   ls="-",  marker="o", ms=3, label="RV actuelle")
        _legend(ax4, fontsize=6)
    ax4.set_xlabel("Fenêtre (jours)")
    ax4.set_ylabel("RV (%)")
    _panel_title(ax4, "RV HISTORIQUE GLD / XAU", "Corrélation ~1 — même série log-returns")
    _badge(ax4, "RV CÔNE", GREEN)

    # ── Tableau récapitulatif ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    headers = ["Instrument", "Spot", "ATM IV 30d", "RR25 30d", "BF25 30d",
               "RV 30d", "IVR 30d", "Signal", "Strikes", "QA Score"]
    rows_table = []
    for key, lbl in [("gld","GLD ETF"),("xauusd","XAU/USD")]:
        d   = data.get(key, {})
        surf  = d.get("surface", [])
        spot  = d.get("spot")
        q     = d.get("quality", {})
        ivr_d = d.get("ivr_ivp", [])
        cone  = d.get("rv_cone", {})
        t30   = next((t for t in surf if abs(t["tenor_days"]-30) < 15), None)
        ivr30 = next((x for x in ivr_d if abs(x["tenor_days"]-30) < 15), None)
        rv30  = cone.get("30", {}).get("current")
        rows_table.append([
            lbl,
            f"${spot:.2f}" if spot else "—",
            f"{t30['atm_iv']:.2f}%" if t30 and t30.get("atm_iv") else "—",
            f"{t30['rr25']:+.2f}%"  if t30 and t30.get("rr25") is not None else "—",
            f"{t30['bf25']:+.2f}%"  if t30 and t30.get("bf25") is not None else "—",
            f"{rv30:.1f}%"          if rv30 else "—",
            f"{ivr30['ivr']:.3f}"   if ivr30 and ivr30.get("ivr") is not None else "—",
            ivr30["signal"]         if ivr30 else "—",
            str(q.get("n_strikes", "—")),
            f"{q.get('score','—')}/100",
        ])
    if rows_table:
        col_w = [0.12, 0.07, 0.08, 0.07, 0.07, 0.07, 0.07, 0.09, 0.08, 0.08]
        col_x = [sum(col_w[:i]) for i in range(len(col_w))]
        row_h = 0.14
        for j, (h, x) in enumerate(zip(headers, col_x)):
            ax5.text(x+0.01, 0.96, h, transform=ax5.transAxes,
                     color=TEXTDIM, fontsize=7, fontweight="bold", va="top")
        for i, row in enumerate(rows_table):
            y = 0.96 - (i+1) * row_h - 0.05
            rect = FancyBboxPatch((0, y-0.01), 1.0, row_h-0.01,
                                   transform=ax5.transAxes,
                                   color=GOLDD if i == 0 else PANEL2, alpha=0.4,
                                   boxstyle="round,pad=0.01", zorder=0)
            ax5.add_patch(rect)
            for j, (val, x) in enumerate(zip(row, col_x)):
                c = TEXT
                if j == 7:
                    c = RED if "SELL" in str(val) else GREEN if "BUY" in str(val) else TEXT
                elif j == 9:
                    try:
                        sv = int(str(val).split("/")[0])
                        c = GREEN if sv >= 80 else ORANGE if sv >= 60 else RED
                    except ValueError:
                        pass
                ax5.text(x+0.01, y+0.05, str(val), transform=ax5.transAxes,
                         color=c, fontsize=7.5, va="center", fontfamily="monospace")
    _panel_title(ax5, "TABLEAU RÉCAPITULATIF — GLD / XAU/USD", "Référence cross: 30d tenor")
    _badge(ax5, "SYNTHÈSE", GOLD)

    fig.text(0.03, 0.012,
             f"GLD ETF ≈ 0.0971 oz or · ratio GC/GLD = {ratio:.4f} · "
             "IV% invariante par scaling · XAU spot = GC=F front-month (basis ~0) · "
             "RV XAU = GLD (corrélation ~1)",
             color=TEXTDIM, fontsize=6.5, fontfamily="monospace")
    return fig


# ─────────────────────────────────────────────────────────────────
#  BAYES — FENÊTRE DE PRIX  [CONFORMITÉ INSTITUTIONNELLE v2]
#
#  Refs primaires:
#    Bakshi, Kapadia, Madan (2003) — BKM model-free moments
#    Malz (2012), Financial Risk Management §6.3-6.4
#    Cornish & Fisher (1937); Joanes & Gill (1998)
#    Carr & Madan (1998/2004) — variance swap pricing
#    BCBS FRTB (2019) §MAR33 — ES/CVaR standard
#    Reiswich & Wystup (2012) — FX smile conventions
#
#  CORRECTIONS v2 vs v1 (audit institutionnel McKinsey/BlackRock):
#    F1 P0: σ_post = √(w·IV² + (1-w)·RV²)  [variance blend, Jensen]
#    F2 P0: γ₂ = 8·BF25/σ_atm²              [Malz §6.4 — σ_atm² pas σ_atm]
#    F3 P1: γ₁ = 6·RR25/σ_atm               [Malz §6.3 — coeff 6 pas 3]
#    F4 P1: V(T)=σ²T monotone enforced       [calendar arbitrage]
#    F5 P1: BKM model-free si ≥4 strikes     [méthode primaire]
#    F6 P1: Vol-of-vol bands p10/p90          [incertitude modèle]
# ─────────────────────────────────────────────────────────────────

# Expense ratio GLD ETF (SPDR Gold Shares prospectus)
GLD_EXPENSE_RATIO = 0.0040   # 40 bps/an

# z-quantiles exacts Φ⁻¹ — intervalles bilatéraux institutionnels
# ±1σ couvre 68.27%; labels précis pour rapports
_Z_VALS = {
    "68.3": 1.0000,   # ±1σ
    "90":   1.6449,
    "95":   1.9600,
    "99":   2.3263,
}
# Clés entières pour compatibilité calculs internes
_Z_INT  = {68: 1.0000, 90: 1.6449, 95: 1.9600, 99: 2.3263}
# z unilatéraux pour VaR downside (négatifs)
_Z_INT_VAR = {0.05: -1.6449, 0.025: -1.9600, 0.01: -2.3263}


# ─────────────────────────────────────────────────────────────────
#  HELPERS STATISTIQUES BAYES
# ─────────────────────────────────────────────────────────────────

def _rv_interp_loglinear(td, rv_cone, key="current"):
    """
    Interpolation log-linéaire de rv_cone[key] au tenor td (jours).
    Log-RV plus régulier que RV pour interpolation terme en terme.
    Retourne val en % annualisé, ou None si données insuffisantes.
    """
    cone_w = sorted([int(k) for k in rv_cone.keys()])
    if not cone_w:
        return None
    rv_pts = [(w, rv_cone[str(w)].get(key, 0.0)) for w in cone_w
              if rv_cone[str(w)].get(key, 0.0) > 0]
    if not rv_pts:
        return None
    xs, ys = [p[0] for p in rv_pts], [p[1] for p in rv_pts]
    if td <= xs[0]:
        return ys[0]
    if td >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= td <= xs[i+1]:
            frac   = (td - xs[i]) / (xs[i+1] - xs[i])
            log_rv = math.log(ys[i]) + frac * (math.log(ys[i+1]) - math.log(ys[i]))
            return math.exp(log_rv)
    return ys[-1]


def _cf_w(z, γ1, γ2):
    """
    Cornish-Fisher quantile expansion — ordre complet O(γ₁²).
    Joanes & Gill (1998); Cornish & Fisher (1937).
    w(z) = z + (z²-1)γ₁/6 + (z³-3z)γ₂/24 − (2z³-5z)γ₁²/36
    """
    return (z
            + (z**2 - 1.0) * γ1 / 6.0
            + (z**3 - 3.0*z) * γ2 / 24.0
            - (2.0*z**3 - 5.0*z) * γ1**2 / 36.0)


def _cf_dw_dz(z, γ1, γ2):
    """
    Dérivée dw/dz — Jacobien pour PDF CF.
    w'(z) = 1 + zγ₁/3 + (3z²-3)γ₂/24 − (6z²-5)γ₁²/36
    """
    return (1.0
            + z * γ1 / 3.0
            + (3.0*z**2 - 3.0) * γ2 / 24.0
            - (6.0*z**2 - 5.0) * γ1**2 / 36.0)


def _cf_pdf_vectorized(S_grid, S0, σT, γ1, γ2, n_z=600):
    """
    PDF de S_T sous log-normale CF, μ=0.
    Transformation de variable rigoureuse:
      R = log(S_T/S₀) ~ CF(0, σT, γ₁, γ₂)
      f_S(s) = φ(z(r)) / (s · σT · |w'(z)|)
    Jacobien complet O(γ₁²). Normalisation numérique.
    """
    if σT < 1e-8 or S0 <= 0:
        return np.zeros(len(S_grid))
    z_grid  = np.linspace(-5.0, 5.0, n_z)
    w_vals  = np.array([_cf_w(z, γ1, γ2) for z in z_grid])
    dw_vals = np.array([_cf_dw_dz(z, γ1, γ2) for z in z_grid])
    S_param = S0 * np.exp(w_vals * σT)
    phi_z   = np.exp(-0.5 * z_grid**2) / math.sqrt(2.0 * math.pi)
    dS_dz   = S_param * σT * dw_vals
    valid   = np.abs(dS_dz) > 1e-10
    pdf_S   = np.where(valid, phi_z / np.abs(dS_dz), 0.0)
    idx         = np.argsort(S_param)
    S_sorted    = S_param[idx]
    pdf_sorted  = pdf_S[idx]
    _, uniq     = np.unique(S_sorted, return_index=True)
    pdf_out = np.maximum(0.0, np.interp(S_grid, S_sorted[uniq], pdf_sorted[uniq],
                                        left=0.0, right=0.0))
    area = _np_trapz(pdf_out, S_grid)
    if area > 1e-12:
        pdf_out /= area
    return pdf_out


def _es_from_pdf(S_grid, pdf, VaR_lo, α):
    """
    Expected Shortfall (CVaR) numérique: ES_α = E[S_T | S_T ≤ VaR_α].
    Réf: BCBS FRTB (2019) §MAR33.
    """
    mask = S_grid <= VaR_lo
    if not mask.any() or α <= 0:
        return VaR_lo
    num = _np_trapz(S_grid[mask] * pdf[mask], S_grid[mask])
    den = _np_trapz(pdf[mask], S_grid[mask])
    return num / den if den > 1e-12 else VaR_lo


def _bkm_moments(smile_list, S, T, r, q):
    """
    BKM (2003) model-free risk-neutral moments depuis smile discret.
    Bakshi, Kapadia, Madan (2003) "Stock Return Characteristics,
    Skew Laws, and the Differential Pricing of Individual Equity Options"
    JF 58(3) — équations (3)-(5).

    Utilise options OTM uniquement: calls K≥F, puts K<F.
    Intégration numérique (trapèze) tronquée aux strikes observés.
    Conservatif: sous-estime légèrement les moments de queue.

    Returns: (γ1, γ2) — skewness et excess kurtosis bornés
             au domaine de validité CF. (None, None) si échec.
    """
    if smile_list is None or len(smile_list) < 4:
        return None, None
    try:
        F = S * math.exp((r - q) * T)
        pts = sorted(smile_list, key=lambda x: x['K'])
        Ks  = np.array([p['K']        for p in pts], dtype=float)
        IVs = np.array([p['iv']/100.  for p in pts], dtype=float)
        if len(Ks) < 4:
            return None, None

        # Prix OTM uniquement (meilleure stabilité numérique)
        prices = np.array([
            max(0.0, bs_price(S, float(K), T, float(iv), r, bool(K >= F)))
            for K, iv in zip(Ks, IVs)
        ])

        k = np.log(Ks / F)   # log-moneyness vs forward

        # Intégrandes BKM (2003) eq 3-5 — OTM unifiés
        # V: variance
        w_V = 2.0 * (1.0 - k) / Ks**2

        # W: troisième moment (skewness)
        # calls K≥F: (6k - 3k²)/K²
        # puts  K<F: -(6k + 3k²)/K²
        w_W = np.where(Ks >= F,
                       (6.0*k - 3.0*k**2) / Ks**2,
                       -(6.0*k + 3.0*k**2) / Ks**2)

        # X: quatrième moment (kurtosis)
        # calls K≥F: (12k² - 4k³)/K²
        # puts  K<F: (12k² + 4k³)/K²
        w_X = np.where(Ks >= F,
                       (12.0*k**2 - 4.0*k**3) / Ks**2,
                       (12.0*k**2 + 4.0*k**3) / Ks**2)

        exp_rT = math.exp(r * T)
        V = exp_rT * _np_trapz(w_V * prices, Ks)
        W = exp_rT * _np_trapz(w_W * prices, Ks)
        X = exp_rT * _np_trapz(w_X * prices, Ks)

        if V <= 1e-12 or not math.isfinite(V):
            return None, None

        # Moments risk-neutral (BKM Props. 1-3)
        μ  = math.exp(r*T) - 1.0 - V/2.0 - W/6.0 - X/24.0
        σ2 = V - μ**2
        if σ2 <= 1e-12:
            return None, None

        skew = (W - 3.0*μ*V + 2.0*μ**3) / σ2**1.5
        kurt = (X - 4.0*μ*W + 6.0*μ**2*V - 3.0*μ**4) / σ2**2 - 3.0

        # Borne au domaine de validité CF (Cornish 1960)
        γ1 = max(-1.5, min(1.5, float(skew)))
        γ2 = max(0.0,  min(6.0, float(kurt)))
        return γ1, γ2

    except Exception:
        return None, None


def _malz_moments(rr25_raw, bf25_raw, iv_dec):
    """
    Approximation de Malz (2012) §6.3-6.4 quand smile complet indisponible.
    Corrections v2 (audit institutionnel):
      γ₁ = 6·RR25_decimal/σ_atm         [coeff 6 — Malz §6.3]
      γ₂ = 8·BF25_decimal/σ_atm²        [σ_atm² — Malz §6.4]
    Convention RR25 = σ_call25Δ − σ_put25Δ
      gold: put>call → RR25<0 → γ₁<0 → queue gauche lourde ✓
    """
    iv_safe = max(iv_dec, 1e-4)
    γ1_raw  = 6.0 * (rr25_raw / 100.0) / iv_safe         if rr25_raw != 0.0 else 0.0
    γ2_raw  = 8.0 * (bf25_raw / 100.0) / iv_safe**2      if bf25_raw != 0.0 else 0.0
    γ2_raw  = max(0.0, γ2_raw)
    return γ1_raw, γ2_raw


# ─────────────────────────────────────────────────────────────────
#  COMPUTE — FENÊTRE BAYÉSIENNE INSTITUTIONNELLE
# ─────────────────────────────────────────────────────────────────
def compute_bayes_windows(surface, rv_cone, S, r=0.05,
                           q=GLD_EXPENSE_RATIO, smiles=None):
    """
    Fenêtre bayésienne de prix — conformité institutionnelle (v2).

    MODÈLE STATISTIQUE:
      Mesure P (historique): μ=0 (martingale)
      Centre: S_0 (mesure P). Forward Q: F_Q = S·exp((r-q)T) [informatif]

    VARIANCE BLEND (F1 corrigé — Jensen):
      σ²_post(T) = w·IV²(T) + (1-w)·RV²(T),  w=0.50
      σ_post = √σ²_post
      RV interpolé log-linéaire entre tenors du cône

    CALENDAR ARBITRAGE (F4):
      V(T) = σ²_post(T)·T → enforcement V monotone croissant
      (Carr-Madan 1998: no-arb ↔ dV/dT ≥ 0)

    MOMENTS CF — méthode primaire: BKM (2003) si smile ≥4 strikes:
      γ₁, γ₂ = BKM model-free (Bakshi-Kapadia-Madan 2003)
    Fallback: Malz (2012) §6.3-6.4 (coefficients corrigés v2):
      γ₁ = 6·RR25/σ_atm                [§6.3, coeff 6]
      γ₂ = 8·BF25/σ_atm²               [§6.4, σ_atm²]
      Domaine CF: γ₁∈[-1.5,1.5], γ₂∈[0,6] (Cornish 1960)

    DISTRIBUTION:
      R ~ CF(0, σT, γ₁, γ₂)
      w(z)=z+(z²-1)γ₁/6+(z³-3z)γ₂/24-(2z³-5z)γ₁²/36

    RISK MEASURES:
      Intervalles: 68.3% (±1σ), 90%, 95%, 99%
      VaR unilatéral: 5%, 2.5%, 1%
      ES (CVaR): ES_5%, ES_2.5%, ES_1% [FRTB Basel IV §MAR33]
      Médiane CF: S·exp(-γ₁·σT/6)  [w(z=0) = -γ₁/6]

    VOL-OF-VOL (F6):
      σ_post_lo = √(w·IV² + (1-w)·RV_p10²)  [scénario bas]
      σ_post_hi = √(w·IV² + (1-w)·RV_p90²)  [scénario stress]
    """
    # ── Index smiles par tenor pour BKM ──────────────────────────
    smile_by_tenor = {}
    if smiles:
        for sm in smiles:
            smile_by_tenor[sm["tenor_days"]] = sm.get("smile", [])

    # ── PASS 1: calcul σ_post pour chaque tenor ───────────────────
    raw_results = []
    for t in surface:
        iv_atm = t.get("atm_iv")
        if iv_atm is None:
            continue
        td = t["tenor_days"]
        T  = max(t["T"], 1.0/365.0)

        # Vol composantes
        iv_dec = iv_atm / 100.0
        rv_raw = _rv_interp_loglinear(td, rv_cone, key="current")
        rv_p10 = _rv_interp_loglinear(td, rv_cone, key="p10")
        rv_p90 = _rv_interp_loglinear(td, rv_cone, key="p90")

        rv_fallback = iv_dec   # si RV manquante → IV seule
        if rv_raw is None or rv_raw <= 0:
            rv_dec = rv_fallback; rv_raw = iv_atm
            rv_p10 = rv_p10 or iv_atm; rv_p90 = rv_p90 or iv_atm
        else:
            rv_dec = rv_raw / 100.0
            rv_p10 = (rv_p10 or rv_raw) / 100.0
            rv_p90 = (rv_p90 or rv_raw) / 100.0

        # F1 CORRIGÉ: variance blend (pas vol levels)
        w_q    = 0.50
        σ2_post = w_q * iv_dec**2 + (1.0 - w_q) * rv_dec**2
        σ_post  = math.sqrt(σ2_post)

        # Vol-of-vol bounds (F6)
        σ_post_lo = math.sqrt(w_q * iv_dec**2 + (1.0-w_q) * rv_p10**2)
        σ_post_hi = math.sqrt(w_q * iv_dec**2 + (1.0-w_q) * rv_p90**2)

        raw_results.append({
            "td": td, "T": T,
            "iv_atm": iv_atm, "rv_raw": rv_raw,
            "iv_dec": iv_dec, "rv_dec": rv_dec,
            "rv_p10": rv_p10, "rv_p90": rv_p90,
            "σ_post": σ_post, "σ_post_lo": σ_post_lo, "σ_post_hi": σ_post_hi,
            "w_q": w_q, "t_obj": t,
        })

    if not raw_results:
        return []

    raw_results.sort(key=lambda x: x["T"])

    # ── PASS 2: Calendar arbitrage — V(T)=σ²T monotone croissant ──
    V_running    = 0.0
    V_running_lo = 0.0
    V_running_hi = 0.0
    for rec in raw_results:
        V_i = rec["σ_post"]**2 * rec["T"]
        V_running = max(V_running, V_i)
        rec["σ_post_cal"] = math.sqrt(V_running / rec["T"])
        rec["cal_arb_adj"] = abs(rec["σ_post_cal"] - rec["σ_post"]) > 1e-6

        # Enforcement indépendant pour chaque bound vol-of-vol
        V_lo = rec["σ_post_lo"]**2 * rec["T"]
        V_hi = rec["σ_post_hi"]**2 * rec["T"]
        V_running_lo = max(V_running_lo, V_lo)
        V_running_hi = max(V_running_hi, V_hi)
        # Garantie lo ≤ centre ≤ hi après enforcement
        V_running_lo = min(V_running_lo, V_running)
        V_running_hi = max(V_running_hi, V_running)
        rec["σ_post_lo"] = math.sqrt(max(V_running_lo, 1e-12) / rec["T"])
        rec["σ_post_hi"] = math.sqrt(max(V_running_hi, 1e-12) / rec["T"])

        rec["σ_post"] = rec["σ_post_cal"]

    # ── PASS 3: moments CF par tenor ─────────────────────────────
    results = []
    for rec in raw_results:
        td, T   = rec["td"], rec["T"]
        t_obj   = rec["t_obj"]
        σ_post  = rec["σ_post"]
        σT      = σ_post * math.sqrt(T)
        iv_dec  = rec["iv_dec"]

        # Choix méthode γ₁/γ₂
        γ1_bkm = γ2_bkm = None
        moment_source = "Malz(2012)"

        # BKM model-free si smile disponible au tenor (±7j)
        best_sm, best_dist = None, 8
        for sm_td, sm_list in smile_by_tenor.items():
            dist = abs(sm_td - td)
            if dist < best_dist and len(sm_list) >= 4:
                # Guard: vérifier que les strikes sont dans le scale de S
                # (évite BKM corrompu si smiles GLD passés avec S=XAU)
                med_K = sorted(sm_list, key=lambda x: x['K'])[len(sm_list)//2]['K']
                if 0.3 * S < med_K < 3.0 * S:
                    best_dist = dist; best_sm = sm_list
        if best_sm is not None:
            γ1_bkm, γ2_bkm = _bkm_moments(best_sm, S, T, r, GLD_EXPENSE_RATIO)
            if γ1_bkm is not None:
                moment_source = f"BKM(2003) @{td}d"

        if γ1_bkm is not None:
            γ1_raw, γ2_raw = γ1_bkm, γ2_bkm
        else:
            # Fallback Malz — coefficients corrigés (F2, F3)
            rr25_raw = t_obj.get("rr25") or 0.0
            bf25_raw = t_obj.get("bf25") or 0.0
            γ1_raw, γ2_raw = _malz_moments(rr25_raw, bf25_raw, iv_dec)

        # Bornage domaine CF (Cornish 1960)
        γ1 = max(-1.5, min(1.5, γ1_raw))
        γ2 = max(0.0,  min(6.0, γ2_raw))

        # Dégénérescence CF: w'(z)≤0 — plage ±5.0 alignée avec _cf_pdf_vectorized z_grid
        cf_degenerate = any(_cf_dw_dz(z, γ1, γ2) <= 0.0
                            for z in np.linspace(-5.0, 5.0, 200))
        if cf_degenerate:
            γ1, γ2 = 0.0, 0.0

        # ── Centre et médiane CF ──────────────────────────────────
        S_center = S   # martingale P; F_Q = S·exp((r-q)T) informatif
        S_median = S_center * math.exp(-γ1 * σT / 6.0)   # w(z=0) = -γ₁/6

        # ── Intervalles VaR bilatéraux ────────────────────────────
        intervals = {}
        for pct, z_abs in sorted(_Z_INT.items()):
            z_hi_cf = _cf_w(+z_abs, γ1, γ2)
            z_lo_cf = _cf_w(-z_abs, γ1, γ2)
            S_hi = S_center * math.exp(z_hi_cf * σT)
            S_lo = S_center * math.exp(z_lo_cf * σT)
            # Garantie lo < center < hi
            S_lo = min(S_lo, S_center * 0.9999)
            S_hi = max(S_hi, S_center * 1.0001)
            intervals[pct] = {
                "lo":     round(S_lo, 2),
                "hi":     round(S_hi, 2),
                "pct_lo": round((S_lo/S - 1.0)*100.0, 2),
                "pct_hi": round((S_hi/S - 1.0)*100.0, 2),
            }

        # Emboîtement strict (68 ⊂ 90 ⊂ 95 ⊂ 99) — epsilon proportionnel au spot
        _eps = S_center * 1e-4   # ~0.03$ sur GLD@300, scale-invariant
        pcts = sorted(intervals.keys())
        for i in range(len(pcts)-1):
            p_in, p_out = pcts[i], pcts[i+1]
            if intervals[p_in]["lo"] < intervals[p_out]["lo"]:
                intervals[p_in]["lo"] = round(intervals[p_out]["lo"] + _eps, 2)
            if intervals[p_in]["hi"] > intervals[p_out]["hi"]:
                intervals[p_in]["hi"] = round(intervals[p_out]["hi"] - _eps, 2)

        # ── VaR unilatéral downside ───────────────────────────────
        _VAR_LEVELS = [(0.05, "5pct"), (0.025, "2_5pct"), (0.01, "1pct")]
        var_down = {}
        for α, key in _VAR_LEVELS:
            z_cf = _cf_w(_Z_INT_VAR[α], γ1, γ2)
            var_down[key] = round(S_center * math.exp(z_cf * σT), 2)

        # ── Expected Shortfall (CVaR) FRTB §MAR33 ────────────────
        margin_es  = max(S_center * 0.20, 4.0 * σT * S_center)
        S_lo_grid  = max(S_center * 0.01, S_center - margin_es)
        S_hi_grid  = S_center + margin_es
        S_pdf_grid = np.linspace(S_lo_grid, S_hi_grid, 800)
        pdf_vals   = _cf_pdf_vectorized(S_pdf_grid, S_center, σT, γ1, γ2)
        es_out = {}
        for α, var_key, es_key in [(0.05,"5pct","ES_5pct"),(0.025,"2_5pct","ES_2_5pct"),(0.01,"1pct","ES_1pct")]:
            VaR_level  = var_down[var_key]
            es_out[es_key] = round(_es_from_pdf(S_pdf_grid, pdf_vals, VaR_level, α), 2)

        # ── Vol-of-vol bounds 68.3% ───────────────────────────────
        σT_lo = rec["σ_post_lo"] * math.sqrt(T)
        σT_hi = rec["σ_post_hi"] * math.sqrt(T)
        vov_68_lo = round(S_center * math.exp(_cf_w(-1.0, γ1, γ2) * σT_lo), 2)
        vov_68_hi = round(S_center * math.exp(_cf_w(+1.0, γ1, γ2) * σT_hi), 2)

        # ── Forward Q-measure (informatif uniquement) ─────────────
        F_Q = round(S * math.exp((r - GLD_EXPENSE_RATIO) * T), 2)

        results.append({
            "tenor_days":      td,
            "T":               round(T, 6),
            # Vol
            "iv_atm":          round(rec["iv_atm"], 3),
            "rv_current":      round(rec["rv_raw"], 3),
            "sigma_post":      round(σ_post * 100.0, 4),
            "sigma_post_lo":   round(rec["σ_post_lo"] * 100.0, 4),
            "sigma_post_hi":   round(rec["σ_post_hi"] * 100.0, 4),
            "sigma_T":         round(σT, 6),
            "cal_arb_adj":     rec["cal_arb_adj"],
            "w_q":             0.50,
            # CF moments
            "gamma1":          round(γ1, 4),
            "gamma2":          round(γ2, 4),
            "cf_degenerate":   cf_degenerate,
            "moment_source":   moment_source,
            # Distribution
            "S_center":        round(S_center, 2),
            "S_median":        round(S_median, 2),
            "F_Q":             F_Q,
            # Risk measures
            "intervals":       intervals,
            "VaR_down":        var_down,
            "ES":              es_out,
            # Vol-of-vol uncertainty
            "vov_68_lo":       vov_68_lo,
            "vov_68_hi":       vov_68_hi,
        })

    results.sort(key=lambda x: x["tenor_days"])
    return results


# ─────────────────────────────────────────────────────────────────
#  PAGE 4 — FENÊTRE BAYÉSIENNE  [INSTITUTIONNEL v2]
# ─────────────────────────────────────────────────────────────────
def page_bayes(data, instrument="gld", page_num=4, total_pages=5):
    instr   = data.get(instrument, data)
    meta    = data.get("meta", {})
    S       = float(instr.get("spot") or data.get("spot", 0) or 0)
    surface = instr.get("surface") or data.get("surface", [])
    rv_cone = instr.get("rv_cone") or data.get("rv_cone", {})
    smiles  = instr.get("smiles") or data.get("smiles", [])
    r       = float(meta.get("rate") or 0.05)

    label_map = {"gld": "GLD ETF", "xauusd": "XAU/USD"}
    ts = meta.get("fetched_at", "")[:19].replace("T", " ")

    windows = compute_bayes_windows(surface, rv_cone, S, r=r, smiles=smiles)

    # Compter tenors BKM vs Malz
    n_bkm   = sum(1 for w in windows if "BKM" in w.get("moment_source",""))
    n_malz  = len(windows) - n_bkm
    bkm_lbl = f"BKM:{n_bkm}  Malz:{n_malz}"

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.text(0.03, 0.978,
             f"GOLD — FENÊTRE BAYÉSIENNE DE PRIX v2 — {label_map.get(instrument, instrument.upper())}",
             color=GOLDL, fontsize=14, fontweight="bold", va="top", fontfamily="monospace")
    fig.text(0.03, 0.958,
             (f"Spot: ${S:.2f}  |  σ_post=√(½IV²+½RV²)  |  μ=0 (martingale P)  |  "
              f"V(T)=σ²T monotone (cal-arb)  |  {bkm_lbl}  |  "
              f"ES: FRTB Basel IV  |  {ts}"),
             color=TEXTDIM, fontsize=7.0, va="top", fontfamily="monospace")
    fig.text(0.97, 0.978, f"PAGE {page_num} / {total_pages}",
             color=TEXTDIM, fontsize=7, va="top", ha="right", fontfamily="monospace")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           top=0.91, bottom=0.07, left=0.05, right=0.97,
                           hspace=0.55, wspace=0.32)

    # ── Fan Chart ─────────────────────────────────────────────────
    ax_fan = fig.add_subplot(gs[:2, :2])

    if windows:
        T_nodes  = np.array([w["T"]             for w in windows])
        σ_nodes  = np.array([w["sigma_post"]/100 for w in windows])
        σ_lo_n   = np.array([w["sigma_post_lo"]/100 for w in windows])
        σ_hi_n   = np.array([w["sigma_post_hi"]/100 for w in windows])
        γ1_nodes = np.array([w["gamma1"]        for w in windows])
        γ2_nodes = np.array([w["gamma2"]        for w in windows])

        # Variance totale — interpolation linéaire (Carr-Madan)
        τ_nodes = σ_nodes**2 * T_nodes
        T_max   = T_nodes[-1]
        T_fine  = np.linspace(1.0/365.0, T_max, 600)
        d_fine  = T_fine * 365.0

        τ_fine  = np.interp(T_fine, T_nodes, τ_nodes)
        σ_fine  = np.sqrt(np.maximum(τ_fine / T_fine, 1e-12))
        γ1_fine = np.interp(T_fine, T_nodes, γ1_nodes)
        γ2_fine = np.interp(T_fine, T_nodes, γ2_nodes)

        # Variance interpolée pour bounds vol-of-vol
        τ_lo_fine = np.interp(T_fine, T_nodes, σ_lo_n**2 * T_nodes)
        τ_hi_fine = np.interp(T_fine, T_nodes, σ_hi_n**2 * T_nodes)
        σ_lo_fine = np.sqrt(np.maximum(τ_lo_fine / T_fine, 1e-12))
        σ_hi_fine = np.sqrt(np.maximum(τ_hi_fine / T_fine, 1e-12))

        def fan_bound(σ_arr, z_sign, z_abs):
            bounds = []
            for i, T in enumerate(T_fine):
                σT_i = σ_arr[i] * math.sqrt(T)
                cf   = _cf_w(z_sign*z_abs, γ1_fine[i], γ2_fine[i])
                bounds.append(S * math.exp(cf * σT_i))
            return np.array(bounds)

        med_arr = np.array([
            S * math.exp(-γ1_fine[i] * σ_fine[i] * math.sqrt(T) / 6.0)
            for i, T in enumerate(T_fine)])

        # Bandes principales
        hi99 = fan_bound(σ_fine, +1, 2.3263); lo99 = fan_bound(σ_fine, -1, 2.3263)
        hi95 = fan_bound(σ_fine, +1, 1.9600); lo95 = fan_bound(σ_fine, -1, 1.9600)
        hi90 = fan_bound(σ_fine, +1, 1.6449); lo90 = fan_bound(σ_fine, -1, 1.6449)
        hi68 = fan_bound(σ_fine, +1, 1.0000); lo68 = fan_bound(σ_fine, -1, 1.0000)

        # Bandes vol-of-vol (incertitude modèle)
        hi68_hi = fan_bound(σ_hi_fine, +1, 1.0000)
        lo68_lo = fan_bound(σ_lo_fine, -1, 1.0000)

        ax_fan.fill_between(d_fine, lo99, hi99, alpha=0.06, color=GOLD, label="99%")
        ax_fan.fill_between(d_fine, lo95, hi95, alpha=0.09, color=GOLD, label="95%")
        ax_fan.fill_between(d_fine, lo90, hi90, alpha=0.13, color=GOLD, label="90%")
        ax_fan.fill_between(d_fine, lo68, hi68, alpha=0.20, color=GOLD, label="68.3% (±1σ)")

        # Vol-of-vol uncertainty envelope (tirets fins)
        ax_fan.fill_between(d_fine, lo68_lo, hi68_hi,
                            alpha=0.05, color=VIOLET, label="VoV p10-p90")
        ax_fan.plot(d_fine, hi68_hi, color=VIOLET, lw=0.8, ls=":", alpha=0.6)
        ax_fan.plot(d_fine, lo68_lo, color=VIOLET, lw=0.8, ls=":", alpha=0.6)

        ax_fan.plot(d_fine, hi99, color=TEXTDIM, lw=0.5, ls=":", alpha=0.5)
        ax_fan.plot(d_fine, lo99, color=TEXTDIM, lw=0.5, ls=":", alpha=0.5)
        ax_fan.plot(d_fine, hi95, color=GOLD,    lw=0.8, ls="--", alpha=0.5)
        ax_fan.plot(d_fine, lo95, color=GOLD,    lw=0.8, ls="--", alpha=0.5)
        ax_fan.plot(d_fine, hi68, color=GOLDL,   lw=1.3, alpha=0.85)
        ax_fan.plot(d_fine, lo68, color=RED,     lw=1.3, alpha=0.85)
        ax_fan.plot(d_fine, med_arr, color=TEXT, lw=1.5, ls="--", label="Médiane CF", alpha=0.8)
        ax_fan.axhline(S, color=GOLD, lw=1.8, label=f"Spot ${S:.2f}")

        # Annotations bornes finales
        for bound, col_a in [(hi99[-1],TEXTDIM),(lo99[-1],TEXTDIM),
                              (hi95[-1],GOLD),(lo95[-1],GOLD),
                              (hi68[-1],GOLDL),(lo68[-1],RED)]:
            ax_fan.annotate(f"${bound:.0f}", (d_fine[-1], bound),
                            textcoords="offset points", xytext=(4, 0),
                            fontsize=5.5, color=col_a, va="center")

        # VaR 5% downside
        for w in windows:
            VaR5 = w["VaR_down"].get("5pct")
            if VaR5:
                ax_fan.scatter(w["tenor_days"], VaR5,
                               color=RED, s=14, zorder=6, alpha=0.8, marker="v")

        ax_fan.set_xlabel("Horizon (jours)")
        ax_fan.set_ylabel(f"Prix {instrument.upper()} ($)")
        _legend(ax_fan, fontsize=6, ncol=3, loc="upper left")

    _panel_title(ax_fan,
                 "FAN CHART BAYÉSIEN — INSTITUTIONNEL v2",
                 "V(T)=σ²T | CF O(γ₁²) | BKM+Malz | 68.3/90/95/99% | ∫VoV p10-p90")
    _badge(ax_fan, "68.3/90/95/99%", GOLD)
    _badge(ax_fan, "VoV BAND", VIOLET, x=0.24)
    _badge(ax_fan, "▼VaR5%", RED, x=0.38)

    # ── σ_post / γ₁ / γ₂ + BKM vs Malz ─────────────────────────
    ax_sig = fig.add_subplot(gs[:2, 2])
    if windows:
        days_w   = [w["tenor_days"]  for w in windows]
        iv_w     = [w["iv_atm"]      for w in windows]
        rv_w     = [w["rv_current"]  for w in windows]
        sp_w     = [w["sigma_post"]  for w in windows]
        sp_lo_w  = [w["sigma_post_lo"] for w in windows]
        sp_hi_w  = [w["sigma_post_hi"] for w in windows]
        g1_w     = [w["gamma1"]      for w in windows]
        g2_w     = [w["gamma2"]      for w in windows]
        src_bkm  = [w.get("moment_source","").startswith("BKM") for w in windows]

        ax_sig.fill_between(days_w, sp_lo_w, sp_hi_w,
                            alpha=0.12, color=VIOLET, label="VoV p10-p90")
        ax_sig.plot(days_w, iv_w, "o-", color=BLUE,  lw=1.5, ms=3, label="IV ATM (Q)")
        ax_sig.plot(days_w, rv_w, "s-", color=GREEN, lw=1.5, ms=3, label="RV log-lin (P)")
        ax_sig.plot(days_w, sp_w, "D-", color=GOLD,  lw=2.2, ms=4, label="σ_post √blend", zorder=5)

        # Marquer tenors BKM
        bkm_d = [d for d, b in zip(days_w, src_bkm) if b]
        bkm_v = [v for v, b in zip(sp_w,   src_bkm) if b]
        if bkm_d:
            ax_sig.scatter(bkm_d, bkm_v, color=GOLDL, s=55,
                           marker="*", zorder=8, label="★BKM", linewidths=0)

        ax2_sig = ax_sig.twinx()
        ax2_sig.plot(days_w, g1_w, "^--", color=RED,    lw=1.2, ms=4, alpha=0.7, label="γ₁ (skew)")
        ax2_sig.plot(days_w, g2_w, "v--", color=ORANGE, lw=1.2, ms=4, alpha=0.7, label="γ₂ (kurt)")
        ax2_sig.axhline(0, color=BORDER, lw=0.5)
        ax2_sig.set_ylabel("γ₁, γ₂", fontsize=6.5, color=TEXTDIM)
        ax2_sig.tick_params(axis="y", labelsize=6, colors=TEXTDIM)

        _legend(ax_sig, fontsize=5.5, loc="upper left")
        ax2_sig.legend(fontsize=5.5, loc="upper right", facecolor=PANEL2,
                       edgecolor=BORDER, labelcolor=TEXT)
        ax_sig.set_xlabel("Tenor (jours)")
        ax_sig.set_ylabel("Vol annualisée (%)")
        all_v = iv_w + rv_w + sp_w
        if all_v:
            m = (max(all_v)-min(all_v))*0.15 + 0.5
            ax_sig.set_ylim(max(0, min(all_v)-m), max(all_v)+m)

    _panel_title(ax_sig, "σ_post / γ₁ / γ₂",
                 "★BKM(2003) model-free | Malz §6.3-6.4 (corr. v2)")
    _badge(ax_sig, "POSTERIOR VOL + CF", VIOLET)

    # ── Distributions CF + ES pour 3 tenors ──────────────────────
    tenor_targets = [30, 60, 90]
    dist_windows  = [next((x for x in windows if abs(x["tenor_days"]-tg) < 25), None)
                     for tg in tenor_targets]

    for col, (w, tgt) in enumerate(zip(dist_windows, tenor_targets)):
        ax_d = fig.add_subplot(gs[2, col])
        if w is None:
            ax_d.axis("off")
            _panel_title(ax_d, f"~{tgt}d — données insuffisantes")
            continue

        σ_post = w["sigma_post"] / 100.0
        T_w    = w["T"]
        σT_w   = σ_post * math.sqrt(T_w)
        γ1_w, γ2_w = w["gamma1"], w["gamma2"]
        S_c    = w["S_center"]
        S_med  = w["S_median"]

        margin = max(S_c * 0.22, 4.5 * σT_w * S_c)
        x_lo   = max(S_c * 0.25, S_c - margin)
        x_hi   = S_c + margin
        S_grid = np.linspace(x_lo, x_hi, 800)

        pdf = _cf_pdf_vectorized(S_grid, S_c, σT_w, γ1_w, γ2_w)
        pdf_max = max(pdf.max(), 1e-12)

        for pct, alpha_f, col_f in [(99,0.06,GOLD),(95,0.10,GOLD),(68,0.22,GOLD)]:
            lo_b = w["intervals"][pct]["lo"]
            hi_b = w["intervals"][pct]["hi"]
            mask = (S_grid >= lo_b) & (S_grid <= hi_b)
            if mask.any():
                ax_d.fill_between(S_grid[mask], pdf[mask], alpha=alpha_f, color=col_f)

        ax_d.plot(S_grid, pdf, color=GOLDL, lw=2.0)
        ax_d.axvline(S_c,   color=GOLD,  lw=1.5, ls="-",  label=f"Spot ${S_c:.0f}")
        ax_d.axvline(S_med, color=TEXT,  lw=1.0, ls="--", label=f"Med ${S_med:.0f}")

        VaR5 = w["VaR_down"].get("5pct")
        ES5  = w["ES"].get("ES_5pct")
        if VaR5:
            ax_d.axvline(VaR5, color=RED, lw=1.2, ls="-.", label=f"VaR5% ${VaR5:.0f}")
        if ES5:
            ax_d.axvline(ES5, color=VIOLET, lw=1.0, ls=":", label=f"ES5% ${ES5:.0f}")
            mask_es = S_grid <= (VaR5 or x_lo)
            if mask_es.any():
                ax_d.fill_between(S_grid[mask_es], pdf[mask_es],
                                  alpha=0.30, color=RED)

        lo68, hi68 = w["intervals"][68]["lo"], w["intervals"][68]["hi"]
        for val, col_a, lbl in [(lo68, RED, f"${lo68:.0f}"), (hi68, GOLDL, f"${hi68:.0f}")]:
            if x_lo < val < x_hi:
                ax_d.axvline(val, color=col_a, lw=0.7, ls=":", alpha=0.7)
                ax_d.text(val, pdf_max*0.82, lbl, color=col_a,
                          fontsize=5.0, ha="center", rotation=90, va="top")

        src_lbl = "BKM" if "BKM" in w.get("moment_source","") else "Malz"
        ax_d.set_xlabel("Prix ($)")
        ax_d.set_ylabel("Densité" if col == 0 else "")
        ax_d.set_xlim(x_lo, x_hi)
        ax_d.set_ylim(bottom=0)
        _legend(ax_d, fontsize=5.0, loc="upper right", ncol=1, framealpha=0.3)
        _panel_title(ax_d,
                     f"~{w['tenor_days']}d  σ={w['sigma_post']:.1f}%  [{src_lbl}]",
                     f"γ₁={γ1_w:.2f}  γ₂={γ2_w:.2f}  CF{'✓' if not w['cf_degenerate'] else '⚠'}")

    fig.text(0.03, 0.012,
             "v2: σ=√(½IV²+½RV²) [Jensen] | γ₁=6·RR25/σ [Malz§6.3] | "
             "γ₂=8·BF25/σ² [Malz§6.4] | BKM(2003) model-free | "
             "V(T)=σ²T cal-arb | VoV=p10/p90 cône | ES FRTB Basel IV",
             color=TEXTDIM, fontsize=5.8, fontfamily="monospace")
    return fig

# ─────────────────────────────────────────────────────────────────
#  RAPPORT PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_report(data, pdf_out=DEFAULT_PDF, instrument="gld"):
    out  = Path(pdf_out)
    ts   = data.get("meta", {}).get("fetched_at", "")[:19].replace("T", " ")
    S_gld = data.get("gld", {}).get("spot") or data.get("spot", 0)
    S_xau = data.get("xauusd", {}).get("spot", 0)
    figs  = []
    try:
        print("Génération page 1 — Term Structure / RV Cône / IVR / Skew...")
        figs.append(page1(data, instrument))
        print("Génération page 2 — Smiles / Density / Greeks...")
        figs.append(page2(data, instrument))
        print("Génération page 3 — Comparaison instruments / Jumps / Synthèse...")
        figs.append(page3(data))
        print("Génération page 4 — Fenêtre Bayésienne GLD...")
        figs.append(page_bayes(data, "gld",    page_num=4, total_pages=5))
        print("Génération page 5 — Fenêtre Bayésienne XAU/USD...")
        figs.append(page_bayes(data, "xauusd", page_num=5, total_pages=5))
        print(f"Export PDF → {out}...")
        with PdfPages(out) as pdf:
            for fig in figs:
                pdf.savefig(fig, facecolor=BG)
            d = pdf.infodict()
            d["Title"]   = f"Gold Volatility Report — {ts}"
            d["Subject"] = f"GLD ${S_gld:.2f} / XAU ${S_xau:.2f} — {ts}"
            d["Author"]  = "price.py — yfinance · GLD options · GC=F"
        png_out = out.with_suffix(".png")
        figs[3].savefig(png_out, dpi=180, facecolor=BG, bbox_inches="tight")
        sz = out.stat().st_size // 1024
        print(f"\nPDF:    {out}  ({sz} KB, {len(figs)} pages)")
        print(f"Aperçu: {png_out}  (page 4 — Bayes GLD)")
    finally:
        plt.close("all")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    t0  = datetime.now(timezone.utc)
    ts  = t0.strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'═'*62}")
    print(f"  GOLD VOLATILITY PIPELINE — {ts}")
    print(f"  Exécution: données en temps réel via yfinance")
    print(f"{'═'*62}")
    try:
        data = run_fetch(json_out=DEFAULT_JSON)
        run_report(data, pdf_out=DEFAULT_PDF, instrument="gld")
    except PipelineError as e:
        print(f"\n[FATAL] {e}")
        raise SystemExit(1) from e
    t1  = datetime.now(timezone.utc)
    elapsed = (t1 - t0).total_seconds()
    S_gld = data.get("gld", {}).get("spot", 0)
    S_xau = data.get("xauusd", {}).get("spot", 0)
    print(f"\n{'═'*62}")
    print(f"  TERMINÉ en {elapsed:.1f}s")
    print(f"  GLD  ${S_gld:.2f}  |  XAU/USD  ${S_xau:.2f}")
    print(f"  JSON : {DEFAULT_JSON}")
    print(f"  PDF  : {DEFAULT_PDF}  (5 pages)")
    print(f"       p1: Term Structure · RV Cône · IVR · Skew")
    print(f"       p2: Smiles · Densité · Greeks")
    print(f"       p3: Comparaison · Sauts · Synthèse")
    print(f"       p4: Fenêtre Bayésienne GLD")
    print(f"       p5: Fenêtre Bayésienne XAU/USD")
    print(f"{'═'*62}\n")

if __name__ == "__main__":
    main()
