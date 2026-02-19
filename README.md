# Gold Volatility Pipeline

**Rapport de volatilité professionnel sur l'or — GLD & XAU/USD — en une seule commande.**

Un outil Python qui récupère les données d'options en temps réel, construit une surface de volatilité complète, et génère un rapport PDF de 5 pages avec fenêtre bayésienne de prix, intervalles de confiance, et mesures de risque institutionnelles.

---

## Ce que ça produit

Un rapport PDF de 5 pages + un fichier JSON de données brutes :

| Page | Contenu |
|------|---------|
| **1** | Term structure ATM IV, cône RV historique, IVR (signal achat/vente vol), forward vol, skew |
| **2** | Smiles par expiration, densité risk-neutral Breeden-Litzenberger, heatmap surface, dollar gamma, vega |
| **3** | Comparaison GLD / XAU/USD, sauts détectés (z > 3σ), tableau récapitulatif |
| **4** | Fenêtre bayésienne GLD — fan chart, intervalles 68/90/95/99%, VaR, Expected Shortfall |
| **5** | Fenêtre bayésienne XAU/USD — même framework, rescalé au spot or |

Le rapport utilise un thème sombre institutionnel, lisible sur écran et en impression.

---

## Installation

```bash
pip install yfinance scipy numpy pandas matplotlib
```

Python 3.9+ requis. Aucune clé API nécessaire.

## Utilisation

```bash
python price.py
```

Deux fichiers sont générés dans le répertoire courant :

- `gold_report.pdf` — rapport complet (5 pages)
- `gold_surface_data.json` — données brutes structurées

Temps d'exécution typique : 30–90 secondes (dépend de la réactivité de Yahoo Finance).

---

## Comment ça marche

### Données

L'outil récupère en temps réel via Yahoo Finance :

- **GLD** — spot + chaîne d'options complète (tous les strikes, toutes les expirations ≤ 400 jours)
- **GC=F** — spot XAU/USD (contrat front-month, proxy du spot or)
- **^GVZ** — indice de volatilité implicite de l'or (CBOE)
- **^IRX** — taux sans risque 13 semaines (T-Bill)

Les options sont filtrées : open interest ≥ 10, spread bid-ask ≤ 30%, strikes entre 55% et 165% du spot. La volatilité implicite est calculée par un solveur bisection + Newton hybride sur le modèle Black-Scholes.

### Surface de volatilité

Pour chaque expiration, l'outil calcule :

- **ATM IV** — interpolation log-linéaire entre les deux strikes encadrant le forward
- **25-delta Risk Reversal** — RR25 = IV(call 25Δ) − IV(put 25Δ), mesure le skew
- **25-delta Butterfly** — BF25 = (IV call + IV put)/2 − ATM, mesure la convexité du smile

### Cône de volatilité réalisée

La volatilité réalisée (rolling std des log-returns × √252) est calculée sur 6 ans d'historique pour des fenêtres de 7 à 180 jours. Le cône montre les percentiles p10/p25/p50/p75/p90 — c'est l'équivalent d'un "est-ce que le marché est cher ou pas" pour la volatilité.

Le ratio **IVR = IV / RV** donne un signal direct :
- IVR > 1.15 → la vol implicite est chère → signal **SELL VOL**
- IVR < 0.85 → la vol implicite est sous-évaluée → signal **BUY VOL**

### Fenêtre bayésienne de prix

C'est le cœur du rapport (pages 4 & 5). Le modèle estime la distribution future du prix de l'or à chaque horizon.

**Volatilité postérieure** — blend quadratique IV/RV :

```
σ_post = √(½ × IV² + ½ × RV²)
```

Ce blend en variance (pas en niveaux) est plus conservateur qu'une moyenne linéaire — propriété mathématique connue sous le nom d'inégalité QM ≥ AM.

**Moments de la distribution** — deux méthodes, par ordre de priorité :

1. **BKM (2003)** — moments model-free calculés directement depuis les prix d'options OTM, sans hypothèse de modèle. Utilisé quand le smile a ≥ 4 strikes. Référence : Bakshi, Kapadia & Madan, *Journal of Finance* 58(3).

2. **Malz (2012)** — approximation depuis le risk reversal et le butterfly quand le smile est trop clairsemé :
   - γ₁ (skewness) = 6 × RR25 / σ_atm
   - γ₂ (excess kurtosis) = 8 × BF25 / σ_atm²

**Distribution** — expansion de Cornish-Fisher à l'ordre O(γ₁²) :

```
w(z) = z + (z²−1)γ₁/6 + (z³−3z)γ₂/24 − (2z³−5z)γ₁²/36
```

Cela permet de capturer l'asymétrie et les queues épaisses observées sur l'or, sans recourir à un modèle paramétrique lourd.

**Mesures de risque** :

- Intervalles bilatéraux : 68.3% (±1σ), 90%, 95%, 99%
- VaR unilatéral downside : 5%, 2.5%, 1%
- Expected Shortfall (CVaR) conforme FRTB Basel IV — "en moyenne, combien on perd dans les 5% pires scénarios"

**Contraintes d'arbitrage** — la variance totale V(T) = σ²×T est forcée monotone croissante. C'est la condition de non-arbitrage calendaire (Carr & Madan, 1998).

---

## Structure du JSON

Le fichier `gold_surface_data.json` contient toutes les données calculées :

```
{
  "meta":    { fetched_at, source, gvz, rate, ratio_gld_xau },
  "gld":     { spot, surface[], smiles[], rv_cone{}, ivr_ivp[], fwd_vols[], quality{} },
  "xauusd":  { spot, surface[], rv_cone{}, ivr_ivp[], fwd_vols[], quality{} }
}
```

Chaque élément de `surface[]` contient : expiry, tenor_days, T, atm_iv, iv_25c, iv_25p, rr25, bf25, n_strikes.

Chaque fenêtre du `rv_cone{}` contient : p10, p25, p50, p75, p90, current.

Le score `quality` (0–100) pénalise les surfaces avec peu de tenors ATM, peu de données de smile, ou des violations de calendar arbitrage.

---

## Contrôle qualité automatique

L'outil inclut un module d'audit (`audit_qa`) qui vérifie :

- Nombre minimum de tenors ATM (≥ 6 optimal, ≥ 4 acceptable)
- Disponibilité des données de smile (RR25 sur ≥ 3 tenors)
- Absence de violations de calendar arbitrage dans la surface brute
- Profondeur du cône RV (≥ 4 fenêtres)

Le score QA est affiché en page 3 du rapport et dans la console.

---

## Suite de tests

Le fichier `audit_loop.py` contient 21 tests automatisés couvrant :

| Test | Vérifie |
|------|---------|
| Put-call parity | Black-Scholes correct à 10⁻¹⁴ |
| IV solver round-trip | Convergence < 10⁻⁴ sur tout le domaine |
| Cornish-Fisher | Expansion et dérivée correctes |
| CF dégénérescence | Détection alignée avec le z-grid PDF |
| PDF normalization | Intégrale = 1 ± 2% |
| ES ≤ VaR | Cohérence Expected Shortfall |
| Malz coefficients | Conformité §6.3 et §6.4 |
| Calendar arb enforcement | V(T) monotone après correction |
| Interval nesting | 68 ⊂ 90 ⊂ 95 ⊂ 99 |
| BKM scale guard | Pas de corruption cross-instrument |
| + 11 autres | Vega ≥ 0, delta ∈ [-1,1], edge cases T=0/σ=0, etc. |

```bash
python audit_loop.py
```

---

## Limites connues

- **Source de données** — Yahoo Finance peut avoir du retard, des trous, ou être temporairement indisponible. L'outil gère les timeouts (30s) et les données manquantes avec des fallbacks.
- **Modèle** — l'expansion de Cornish-Fisher dégénère pour des skewness/kurtosis extrêmes. Dans ce cas, le modèle revient à une log-normale standard (γ₁ = γ₂ = 0), signalé par "CF⚠" dans le rapport.
- **XAU/USD** — la surface d'options est celle de GLD rescalée (pas d'options XAU directement tradées sur Yahoo Finance). Les moments BKM ne sont pas disponibles pour XAU (échelle de strikes incompatible), le fallback Malz est systématiquement utilisé.
- **Pas de dividendes** — GLD n'en verse pas, mais le expense ratio (40 bps/an) est pris en compte dans le forward.

---

## Références

- Bakshi, G., Kapadia, N., & Madan, D. (2003). *Stock Return Characteristics, Skew Laws, and the Differential Pricing of Individual Equity Options.* Journal of Finance, 58(3).
- Malz, A. (2012). *Financial Risk Management: Models, History, and Institutions.* Wiley, §6.3–6.4.
- Cornish, E.A. & Fisher, R.A. (1937). *Moments and Cumulants in the Specification of Distributions.* Revue de l'Institut International de Statistique.
- Carr, P. & Madan, D. (1998). *Towards a Theory of Volatility Trading.* Risk Books.
- Breeden, D. & Litzenberger, R. (1978). *Prices of State-Contingent Claims Implicit in Option Prices.* Journal of Business.
- BCBS (2019). *Minimum Capital Requirements for Market Risk.* Basel III FRTB, §MAR33.

---

## Licence

Usage personnel et éducatif.
Gaetan PRUVOT 2026
