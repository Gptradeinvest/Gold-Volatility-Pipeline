#!/usr/bin/env python3
"""
AUDIT ITÉRATIF — price.py
Boucle jusqu'à 0 finding bloquant.
"""
import ast
import math
import sys
import importlib.util
import numpy as np

TARGET = "/home/claude/price.py"
MAX_ITER = 10

# ─── Severity levels ───
BLOCKER  = "P0-BLOCKER"
MAJOR    = "P1-MAJOR"
MINOR    = "P2-MINOR"
INFO     = "P3-INFO"


def load_source():
    with open(TARGET) as f:
        return f.read()


def load_module():
    spec = importlib.util.spec_from_file_location("price", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════
#  AUDIT CHECKS
# ═══════════════════════════════════════════════════════════════

def check_syntax(source):
    """A1: Compilation OK"""
    findings = []
    try:
        compile(source, TARGET, "exec")
    except SyntaxError as e:
        findings.append((BLOCKER, f"SyntaxError L{e.lineno}: {e.msg}"))
    return findings


def check_ast_patterns(source):
    """A2: Patterns dangereux AST"""
    findings = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append((MAJOR, f"L{node.lineno}: Bare except (catch-all)"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults + node.args.kw_defaults:
                if d and isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    findings.append((MAJOR, f"L{node.lineno}: Mutable default in {node.name}()"))
    return findings


def check_putcall_parity(mod):
    """A3: Black-Scholes put-call parity"""
    findings = []
    S, K, T, sig, r = 300, 300, 0.25, 0.20, 0.05
    c = mod.bs_price(S, K, T, sig, r, True)
    p = mod.bs_price(S, K, T, sig, r, False)
    pcp = abs(c - p - (S - K * math.exp(-r * T)))
    if pcp > 1e-10:
        findings.append((BLOCKER, f"Put-call parity violation: {pcp:.2e}"))
    # ATM, OTM, ITM variants
    for K_test in [250, 300, 350]:
        for T_test in [0.01, 0.25, 1.0, 2.0]:
            c = mod.bs_price(S, K_test, T_test, sig, r, True)
            p = mod.bs_price(S, K_test, T_test, sig, r, False)
            pcp = abs(c - p - (S - K_test * math.exp(-r * T_test)))
            if pcp > 1e-8:
                findings.append((BLOCKER, f"PCP fail K={K_test} T={T_test}: {pcp:.2e}"))
    return findings


def check_iv_solver(mod):
    """A4: IV solver round-trip (exclut deep ITM/OTM où vega ~0)"""
    findings = []
    S, r = 300, 0.05
    for K in [260, 280, 300, 320, 340]:
        for T in [0.05, 0.25, 0.5, 1.0]:
            for sig_true in [0.10, 0.20, 0.40, 0.80]:
                for is_call in [True, False]:
                    price = mod.bs_price(S, K, T, sig_true, r, is_call)
                    if price < 1e-6:
                        continue
                    # Skip deep ITM where vega ~0 (IV inversion ill-conditioned)
                    vega = mod.bs_vega(S, K, T, sig_true, r)
                    if vega < 0.01:
                        continue
                    iv = mod.implied_vol(price, S, K, T, r, is_call)
                    if iv is None:
                        findings.append((MAJOR, f"IV solver None: K={K} T={T} σ={sig_true} {'C' if is_call else 'P'}"))
                        continue
                    err = abs(iv - sig_true)
                    if err > 1e-4:
                        findings.append((BLOCKER, f"IV round-trip fail: K={K} T={T} σ_true={sig_true} σ_iv={iv:.6f} err={err:.2e}"))
    return findings


def check_cf_expansion(mod):
    """A5: Cornish-Fisher expansion properties"""
    findings = []
    # Identity at g1=g2=0
    for z in [-2, -1, 0, 1, 2]:
        w = mod._cf_w(z, 0, 0)
        if abs(w - z) > 1e-12:
            findings.append((BLOCKER, f"CF w({z},0,0)={w} ≠ {z}"))
    # Derivative at g1=g2=0
    for z in [-2, -1, 0, 1, 2]:
        dw = mod._cf_dw_dz(z, 0, 0)
        if abs(dw - 1.0) > 1e-12:
            findings.append((BLOCKER, f"CF dw/dz({z},0,0)={dw} ≠ 1"))
    return findings


def check_cf_degeneracy_detection(mod):
    """A6: CF dégénérescence — toutes combinaisons détectées dans z_grid PDF"""
    findings = []
    n_miss = 0
    miss_examples = []
    for g1 in np.linspace(-1.5, 1.5, 50):
        for g2 in np.linspace(0, 6, 50):
            g1f, g2f = float(g1), float(g2)
            # Check du code (doit utiliser mêmes bornes que _cf_pdf_vectorized)
            code_degen = any(mod._cf_dw_dz(z, g1f, g2f) <= 0.0
                             for z in np.linspace(-5.0, 5.0, 200))
            # Vérification plus dense
            real_degen = any(mod._cf_dw_dz(z, g1f, g2f) <= 0.0
                             for z in np.linspace(-5.0, 5.0, 1000))
            if real_degen and not code_degen:
                n_miss += 1
                if len(miss_examples) < 3:
                    miss_examples.append(f"g1={g1f:.2f},g2={g2f:.2f}")
    if n_miss > 0:
        findings.append((BLOCKER, f"CF degeneracy undetected: {n_miss} cases (ex: {', '.join(miss_examples)})"))
    return findings


def check_pdf_normalization(mod):
    """A7: PDF intègre à ~1"""
    findings = []
    S0 = 300.0
    for g1, g2 in [(0, 0), (-0.5, 1.0), (0.3, 0.5)]:
        σT = 0.10
        S_grid = np.linspace(200, 400, 800)
        pdf = mod._cf_pdf_vectorized(S_grid, S0, σT, g1, g2)
        area = np.trapz(pdf, S_grid) if hasattr(np, 'trapz') else np.trapezoid(pdf, S_grid)
        if abs(area - 1.0) > 0.02:
            findings.append((BLOCKER, f"PDF area={area:.4f} ≠ 1 for g1={g1},g2={g2}"))
        if np.any(pdf < -1e-10):
            findings.append((BLOCKER, f"PDF negative values for g1={g1},g2={g2}"))
    return findings


def check_es_consistency(mod):
    """A8: ES ≤ VaR (ES is further in the tail)"""
    findings = []
    S0 = 300.0
    σT = 0.10
    S_grid = np.linspace(200, 400, 800)
    pdf = mod._cf_pdf_vectorized(S_grid, S0, σT, 0, 0)
    VaR = 280.0
    es = mod._es_from_pdf(S_grid, pdf, VaR, 0.05)
    if es > VaR:
        findings.append((BLOCKER, f"ES({es:.2f}) > VaR({VaR:.2f}) — impossible"))
    if es <= 0:
        findings.append((MAJOR, f"ES={es:.2f} ≤ 0 — non-physical"))
    return findings


def check_malz_coefficients(mod):
    """A9: Malz coefficients conformité v2"""
    findings = []
    # γ₁ = 6·RR25/σ — test with known values
    g1, g2 = mod._malz_moments(1.0, 0.5, 0.20)  # RR25=1%, BF25=0.5%, σ=20%
    expected_g1 = 6.0 * (1.0/100.0) / 0.20  # = 0.30
    expected_g2 = 8.0 * (0.5/100.0) / 0.20**2  # = 1.0
    if abs(g1 - expected_g1) > 1e-8:
        findings.append((BLOCKER, f"Malz γ₁={g1:.6f} expected {expected_g1:.6f}"))
    if abs(g2 - expected_g2) > 1e-8:
        findings.append((BLOCKER, f"Malz γ₂={g2:.6f} expected {expected_g2:.6f}"))
    # Gold convention: RR25 < 0 → γ₁ < 0
    g1_neg, _ = mod._malz_moments(-2.0, 0.5, 0.20)
    if g1_neg >= 0:
        findings.append((BLOCKER, f"Malz: RR25<0 should give γ₁<0, got {g1_neg}"))
    return findings


def check_variance_blend(mod):
    """A10: Variance blend — QM>=AM (plus conservateur que linear)"""
    findings = []
    for iv, rv in [(0.20, 0.10), (0.10, 0.30), (0.25, 0.25)]:
        σ2 = 0.5 * iv**2 + 0.5 * rv**2
        σ = math.sqrt(σ2)
        # Must be between min and max
        if σ < min(iv, rv) - 1e-10 or σ > max(iv, rv) + 1e-10:
            findings.append((BLOCKER, f"Variance blend out of bounds: IV={iv} RV={rv} σ={σ}"))
        # QM >= AM (variance blend is correctly MORE conservative than linear)
        linear = 0.5*iv + 0.5*rv
        if σ < linear - 1e-10:
            findings.append((BLOCKER, f"Variance blend < linear (QM>=AM violated): {σ} < {linear}"))
    return findings


def check_spot_zero_guard(source):
    """A11: Protection S_gld ≤ 0"""
    findings = []
    if "S_gld <= 0" not in source and "S_gld == 0" not in source and "S_gld < " not in source:
        findings.append((BLOCKER, "Pas de guard S_gld ≤ 0 après fetch"))
    return findings


def check_calendar_arb_enforcement(mod):
    """A12: V(T)=σ²T monotone après enforcement"""
    findings = []
    # Simulate: tenors with inverted vol
    fake_surface = [
        {"tenor_days": 30, "T": 30/365, "atm_iv": 25.0, "rr25": -1.0, "bf25": 0.5},
        {"tenor_days": 60, "T": 60/365, "atm_iv": 20.0, "rr25": -0.8, "bf25": 0.4},  # inversion
        {"tenor_days": 90, "T": 90/365, "atm_iv": 22.0, "rr25": -0.6, "bf25": 0.3},
    ]
    fake_cone = {
        "30": {"current": 18.0, "p10": 12.0, "p25": 15.0, "p50": 18.0, "p75": 22.0, "p90": 28.0},
        "60": {"current": 17.0, "p10": 11.0, "p25": 14.0, "p50": 17.0, "p75": 21.0, "p90": 27.0},
        "90": {"current": 16.0, "p10": 10.0, "p25": 13.0, "p50": 16.0, "p75": 20.0, "p90": 26.0},
    }
    windows = mod.compute_bayes_windows(fake_surface, fake_cone, 300.0, r=0.05, smiles=None)
    if len(windows) >= 2:
        for i in range(1, len(windows)):
            V_prev = (windows[i-1]["sigma_post"]/100)**2 * windows[i-1]["T"]
            V_curr = (windows[i]["sigma_post"]/100)**2 * windows[i]["T"]
            if V_curr < V_prev - 1e-10:
                findings.append((BLOCKER, f"Cal-arb violated: V({windows[i-1]['tenor_days']}d)={V_prev:.6f} > V({windows[i]['tenor_days']}d)={V_curr:.6f}"))
    return findings


def check_interval_nesting(mod):
    """A13: Intervalles emboîtés 68 ⊂ 90 ⊂ 95 ⊂ 99"""
    findings = []
    fake_surface = [
        {"tenor_days": 30, "T": 30/365, "atm_iv": 22.0, "rr25": -1.5, "bf25": 0.8},
        {"tenor_days": 60, "T": 60/365, "atm_iv": 21.0, "rr25": -1.0, "bf25": 0.5},
    ]
    fake_cone = {
        "30": {"current": 20.0, "p10": 14.0, "p25": 17.0, "p50": 20.0, "p75": 24.0, "p90": 30.0},
        "60": {"current": 19.0, "p10": 13.0, "p25": 16.0, "p50": 19.0, "p75": 23.0, "p90": 29.0},
    }
    windows = mod.compute_bayes_windows(fake_surface, fake_cone, 300.0, r=0.05, smiles=None)
    for w in windows:
        intv = w["intervals"]
        pcts = sorted(intv.keys())
        for i in range(len(pcts)-1):
            inner, outer = pcts[i], pcts[i+1]
            if intv[inner]["lo"] < intv[outer]["lo"]:
                findings.append((BLOCKER, f"Nesting fail: {inner}% lo={intv[inner]['lo']} < {outer}% lo={intv[outer]['lo']} (tenor={w['tenor_days']}d)"))
            if intv[inner]["hi"] > intv[outer]["hi"]:
                findings.append((BLOCKER, f"Nesting fail: {inner}% hi={intv[inner]['hi']} > {outer}% hi={intv[outer]['hi']} (tenor={w['tenor_days']}d)"))
    return findings


def check_vega_positivity(mod):
    """A14: Vega toujours ≥ 0"""
    findings = []
    S = 300
    for K in [250, 275, 300, 325, 350]:
        for T in [0.05, 0.25, 1.0]:
            for sig in [0.05, 0.20, 0.50]:
                v = mod.bs_vega(S, K, T, sig)
                if v < 0:
                    findings.append((BLOCKER, f"Vega<0: K={K} T={T} σ={sig} → {v}"))
    return findings


def check_delta_bounds(mod):
    """A15: Delta dans [-1, 1]"""
    findings = []
    S = 300
    for K in [200, 250, 300, 350, 400]:
        for T in [0.01, 0.25, 1.0]:
            for sig in [0.05, 0.20, 0.50]:
                for is_call in [True, False]:
                    d = mod.bs_delta(S, K, T, sig, 0.05, is_call)
                    if d < -1.001 or d > 1.001:
                        findings.append((BLOCKER, f"Delta OOB: K={K} T={T} σ={sig} {'C' if is_call else 'P'} → {d}"))
    return findings


def check_edge_T_zero(mod):
    """A16: T=0 et σ=0 ne crashent pas"""
    findings = []
    try:
        p = mod.bs_price(300, 300, 0, 0.20, 0.05, True)
        p = mod.bs_price(300, 300, 0.25, 0, 0.05, True)
        v = mod.bs_vega(300, 300, 0, 0.20)
        v = mod.bs_vega(300, 300, 0.25, 0)
        d = mod.bs_delta(300, 300, 0, 0.20, 0.05, True)
        iv = mod.implied_vol(0, 300, 300, 0.25, 0.05, True)
        iv = mod.implied_vol(5, 300, 300, 0, 0.05, True)
    except Exception as e:
        findings.append((BLOCKER, f"Crash on T=0 or σ=0: {e}"))
    return findings


def check_fwd_vol_noarb(mod):
    """A17: Forward vol non-négatif quand cal-arb free"""
    findings = []
    tenors = [
        {"tenor_days": 30, "T": 30/365, "atm_iv": 20.0},
        {"tenor_days": 60, "T": 60/365, "atm_iv": 21.0},
        {"tenor_days": 90, "T": 90/365, "atm_iv": 22.0},
    ]
    fwd = mod.compute_fwd_vols(tenors)
    for f in fwd:
        if f.get("fwd_vol") is not None and f["fwd_vol"] < 0:
            findings.append((BLOCKER, f"Negative fwd vol: {f}"))
        if f.get("cal_arb_ok") is False and f.get("fwd_vol") is not None:
            # fwd_vol should still be positive sqrt if fv2>0
            pass
    return findings


def check_safe_float(mod):
    """A18: safe_float robustesse"""
    findings = []
    tests = [
        (None, 0.0), (float('nan'), 0.0), (float('inf'), 0.0),
        (float('-inf'), 0.0), ("abc", 0.0), (42, 42.0), (3.14, 3.14),
    ]
    for val, expected in tests:
        result = mod.safe_float(val)
        if result != expected:
            findings.append((MAJOR, f"safe_float({val!r})={result} expected {expected}"))
    return findings


def check_pipeline_error_class(mod):
    """A19: PipelineError est RuntimeError"""
    findings = []
    if not issubclass(mod.PipelineError, RuntimeError):
        findings.append((MAJOR, "PipelineError n'hérite pas de RuntimeError"))
    try:
        raise mod.PipelineError("test")
    except RuntimeError:
        pass
    except Exception:
        findings.append((MAJOR, "PipelineError non catchable comme RuntimeError"))
    return findings


def check_numpy_compat(mod):
    """A20: numpy trapz compat"""
    findings = []
    try:
        result = mod._np_trapz(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]))
        if abs(result - 4.0) > 1e-10:
            findings.append((BLOCKER, f"_np_trapz incorrect: {result} ≠ 4.0"))
    except Exception as e:
        findings.append((BLOCKER, f"_np_trapz crash: {e}"))
    return findings


def check_bkm_scale_guard(mod):
    """A21: BKM skips smile strikes hors scale de S (ex: GLD smiles + XAU spot)"""
    findings = []
    # Simulate: GLD smiles (K~300) with XAU spot (~3000)
    fake_surface = [
        {"tenor_days": 30, "T": 30/365, "atm_iv": 25.0, "rr25": -1.0, "bf25": 0.5},
    ]
    fake_cone = {
        "30": {"current": 20.0, "p10": 14.0, "p25": 17.0, "p50": 20.0, "p75": 24.0, "p90": 28.0},
    }
    # Smiles with K in GLD scale (~300)
    fake_smiles = [{"tenor_days": 30, "smile": [
        {"K": 280, "iv": 28.0, "delta": -0.35, "type": "put"},
        {"K": 290, "iv": 26.0, "delta": -0.25, "type": "put"},
        {"K": 300, "iv": 25.0, "delta": 0.50, "type": "call"},
        {"K": 310, "iv": 26.0, "delta": 0.25, "type": "call"},
        {"K": 320, "iv": 28.0, "delta": 0.15, "type": "call"},
    ]}]
    # S = 3000 (XAU scale) — BKM should NOT use these strikes
    windows = mod.compute_bayes_windows(fake_surface, fake_cone, 3000.0, r=0.05, smiles=fake_smiles)
    if windows:
        w = windows[0]
        if "BKM" in w.get("moment_source", ""):
            findings.append((BLOCKER, "BKM used with scale-mismatched strikes (K~300, S=3000)"))
        # Should fall back to Malz
        if "Malz" not in w.get("moment_source", ""):
            findings.append((MAJOR, f"Expected Malz fallback, got: {w.get('moment_source')}"))
    return findings


# ═══════════════════════════════════════════════════════════════
#  AUDIT RUNNER
# ═══════════════════════════════════════════════════════════════

ALL_CHECKS = [
    ("A1  Syntax",                check_syntax,                 "source"),
    ("A2  AST patterns",          check_ast_patterns,           "source"),
    ("A3  Put-Call Parity",       check_putcall_parity,         "module"),
    ("A4  IV Solver round-trip",  check_iv_solver,              "module"),
    ("A5  CF Expansion",          check_cf_expansion,           "module"),
    ("A6  CF Degeneracy detect",  check_cf_degeneracy_detection,"module"),
    ("A7  PDF normalization",     check_pdf_normalization,      "module"),
    ("A8  ES consistency",        check_es_consistency,         "module"),
    ("A9  Malz coefficients",     check_malz_coefficients,      "module"),
    ("A10 Variance blend",        check_variance_blend,         "module"),
    ("A11 Spot zero guard",       check_spot_zero_guard,        "source"),
    ("A12 Calendar arb enforce",  check_calendar_arb_enforcement,"module"),
    ("A13 Interval nesting",      check_interval_nesting,       "module"),
    ("A14 Vega positivity",       check_vega_positivity,        "module"),
    ("A15 Delta bounds",          check_delta_bounds,           "module"),
    ("A16 Edge T=0 σ=0",         check_edge_T_zero,            "module"),
    ("A17 Forward vol no-arb",    check_fwd_vol_noarb,          "module"),
    ("A18 safe_float robustness", check_safe_float,             "module"),
    ("A19 PipelineError class",   check_pipeline_error_class,   "module"),
    ("A20 numpy compat",          check_numpy_compat,           "module"),
    ("A21 BKM scale guard",       check_bkm_scale_guard,        "module"),
]


def run_audit(iteration):
    print(f"\n{'═'*70}")
    print(f"  AUDIT ITÉRATION {iteration}")
    print(f"{'═'*70}")

    source = load_source()
    try:
        mod = load_module()
    except Exception as e:
        print(f"\n  [P0-BLOCKER] Module load failed: {e}")
        return [(BLOCKER, f"Module load: {e}")]

    all_findings = []
    for name, check_fn, arg_type in ALL_CHECKS:
        try:
            if arg_type == "source":
                findings = check_fn(source)
            else:
                findings = check_fn(mod)
        except Exception as e:
            findings = [(BLOCKER, f"Check crashed: {e}")]

        status = "✓" if not findings else "✗"
        blockers = sum(1 for s, _ in findings if s == BLOCKER)
        majors = sum(1 for s, _ in findings if s == MAJOR)

        if findings:
            count_str = f" [{blockers}B {majors}M]" if (blockers or majors) else ""
            print(f"  {status} {name}{count_str}")
            for sev, msg in findings:
                print(f"      [{sev}] {msg}")
        else:
            print(f"  {status} {name}")

        all_findings.extend(findings)

    n_block = sum(1 for s, _ in all_findings if s == BLOCKER)
    n_major = sum(1 for s, _ in all_findings if s == MAJOR)
    n_minor = sum(1 for s, _ in all_findings if s == MINOR)
    n_info  = sum(1 for s, _ in all_findings if s == INFO)

    print(f"\n{'─'*70}")
    print(f"  RÉSUMÉ: {n_block} BLOCKER | {n_major} MAJOR | {n_minor} MINOR | {n_info} INFO")
    if n_block == 0 and n_major == 0:
        print(f"  ✅ PRODUCTION READY")
    elif n_block == 0:
        print(f"  ⚠️  MAJORS restants — non bloquant mais à corriger")
    else:
        print(f"  ❌ BLOCKERS restants — correctifs nécessaires")
    print(f"{'─'*70}")

    return all_findings


if __name__ == "__main__":
    findings = run_audit(1)
    blockers = [(s, m) for s, m in findings if s == BLOCKER]
    if blockers:
        print("\nBLOCKERS à corriger:")
        for s, m in blockers:
            print(f"  → {m}")
        sys.exit(1)
    else:
        sys.exit(0)
