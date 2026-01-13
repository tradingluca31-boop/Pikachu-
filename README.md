# Pikachu - RL Trading Gold FTMO

Agent de Reinforcement Learning pour trader l'or (XAUUSD) et passer le challenge FTMO.

## Objectif

Creer un agent SAC (Soft Actor-Critic) capable de :
- Trader XAUUSD avec un win rate de 40-60%
- Respecter les regles FTMO (max DD 10%, daily DD 5%)
- Generer 10% de profit par cycle de 15 jours

---

## Specifications Techniques

### Donnees

| Element | Valeur |
|---------|--------|
| Asset | XAUUSD (Gold) |
| Periode | 2015-2026 |
| Sources | MetaTrader 5 + Yahoo + FRED + CFTC + Barchart |
| Timeframes | D1, H4, H1, M15 (Multi-TF) |
| Split | Train 70% (2015-2022) / Val 15% (2022-2024) / Test 15% (2024-2026) |

### Algorithme

| Element | Valeur |
|---------|--------|
| Algorithme | SAC (Soft Actor-Critic) |
| Framework | Stable-Baselines3 |
| Action Space | Continu [-1, +1] discretise en 3 niveaux |
| Observation | ~1200-1800 features (Option B: features + lags) |

### Position Sizing (Fixed Fractional)

| Niveau | Risk % |
|--------|--------|
| Small | 0.33% |
| Medium | 0.66% |
| Full | 1.00% |

**Max risk par trade : 1%**

### Risk Management

| Element | Valeur |
|---------|--------|
| SL | Fixe 1.5 ATR |
| TP | Fixe 4.5 ATR (ratio 1:3) |
| Break-Even | A 2R (SL -> entry price) |
| Protection | Risque /2 apres 6 pertes consecutives |
| Positions simultanees | 1 seule |
| Trades/semaine | 2-5 (selectif) |

---

## Features (~1200-1800)

### Architecture Multi-Timeframe

```
D1  --> Tendance majeure + contexte macro
H4  --> Filtre tendance (EMA 20) + structure
H1  --> Signaux d'entree/sortie
M15 --> Timing sniper
```

### Filtres Obligatoires

| Filtre | Condition | Action |
|--------|-----------|--------|
| EMA 20 H4 | Prix > EMA 20 | LONG seulement |
| EMA 20 H4 | Prix < EMA 20 | SHORT seulement |
| Session | London ou NY | Autorise trading |
| Session | Asia | BLOQUE trading |
| News | High impact < 30min | BLOQUE trading |
| Friday | > 20h | BLOQUE nouvelles positions |

---

## FEATURES INSTITUTIONNELLES (Sources: Fed, PIMCO, World Gold Council)

### 17. Real Interest Rates - DRIVER #1 (PIMCO, Fed Chicago)

**Source: FRED API (Gratuit, Daily)**

| Feature | Description | Ticker FRED |
|---------|-------------|-------------|
| `tips_10y_yield` | TIPS 10 ans yield (taux reels) | DFII10 |
| `tips_5y_yield` | TIPS 5 ans yield | DFII5 |
| `tips_10y_change_5d` | Changement TIPS 5 jours | Calcule |
| `tips_10y_change_20d` | Changement TIPS 20 jours | Calcule |
| `real_rate_negative` | 1 si taux reels < 0 (bullish gold) | Calcule |
| `real_rate_above_2pct` | 1 si > 2% (bearish gold) | Calcule |
| `real_rate_regime` | -1/0/+1 selon niveau | Calcule |
| `real_rate_momentum` | Tendance des taux reels | Calcule |

**Interpretation (PIMCO):**
- Correlation Gold/TIPS: **-0.85**
- Real rates baissent 100bps = Gold +8-15%
- Real rates > 2% = Headwind majeur pour Gold

### 18. Inflation Expectations

**Source: FRED API (Gratuit, Daily)**

| Feature | Description | Ticker FRED |
|---------|-------------|-------------|
| `breakeven_10y` | Breakeven inflation 10Y | T10YIE |
| `breakeven_5y` | Breakeven inflation 5Y | T5YIE |
| `breakeven_5y5y` | Forward inflation 5Y5Y | T5YIFR |
| `inflation_change_5d` | Changement breakeven 5j | Calcule |
| `inflation_trend` | Tendance inflation 20j | Calcule |
| `inflation_surprise` | CPI actual vs expected | Calendrier |
| `inflation_above_target` | 1 si > 2% | Calcule |

### 19. Fed Policy (Fed Chicago Research)

**Source: FRED API (Gratuit)**

| Feature | Description | Ticker/Source |
|---------|-------------|---------------|
| `fed_funds_rate` | Taux Fed actuel | FEDFUNDS |
| `fed_funds_upper` | Upper bound | DFEDTARU |
| `fed_rate_expected_3m` | Taux attendu 3 mois | Fed Funds Futures |
| `fed_rate_expected_12m` | Taux attendu 12 mois | Fed Funds Futures |
| `rate_cut_probability` | Proba cut prochain FOMC | CME FedWatch |
| `fed_balance_sheet` | Bilan Fed (total assets) | WALCL |
| `fed_balance_change` | Changement bilan Fed | Calcule |
| `qe_regime` | 1 si QE/expansion, -1 si QT | Calcule |
| `monetary_policy_stance` | Dovish/Neutral/Hawkish | Calcule |

### 20. Gold ETF Flows (World Gold Council) - SIGNAL INSTITUTIONNEL

**Source: Yahoo Finance + SPDR (Gratuit, Daily)**

| Feature | Description | Source |
|---------|-------------|--------|
| `gld_holdings_tonnes` | Holdings GLD en tonnes | SPDR website |
| `gld_holdings_change_1d` | Flows 1 jour | Calcule |
| `gld_holdings_change_5d` | Flows 5 jours | Calcule |
| `gld_holdings_change_20d` | Flows 20 jours | Calcule |
| `gld_flow_momentum` | Tendance des flows | Calcule |
| `gld_flow_acceleration` | Acceleration des flows | Calcule |
| `iau_holdings` | Holdings IAU (iShares) | Yahoo |
| `etf_flow_signal` | Signal agrege ETF flows | Calcule |
| `gld_holdings_zscore` | Z-score holdings 252j | Calcule |
| `etf_flow_direction` | Direction des flows (sign) | Calcule |

**Interpretation:**
- Flows positifs = Demande institutionnelle (Bullish)
- Flows negatifs = Distribution institutionnelle (Bearish)
- Acceleration = Changement de sentiment
- **Correlation R² = 0.85** entre GLD Holdings et prix Gold sur 10 ans

**SIGNAUX DE DIVERGENCE (Tres puissants):**

| Signal | Condition | Interpretation |
|--------|-----------|----------------|
| Divergence Bullish | Prix baisse + ETF inflows | Institutions accumulent = LONG |
| Divergence Bearish | Prix monte + ETF outflows | Institutions distribuent = SHORT |
| Confirmation Bullish | Prix monte + ETF inflows | Trend confirme = Hold LONG |
| Confirmation Bearish | Prix baisse + ETF outflows | Trend confirme = Hold SHORT |

```python
# Feature: Divergence Prix/Flows
def calculate_divergence(price_change_5d, etf_flow_5d):
    price_direction = np.sign(price_change_5d)
    flow_direction = np.sign(etf_flow_5d)
    
    if price_direction != flow_direction:
        return flow_direction  # Divergence: suivre les institutions
    return 0  # Pas de divergence
```

### 21. Central Bank Reserves (World Gold Council)

**Source: World Gold Council (Gratuit, Monthly)**

| Feature | Description |
|---------|-------------|
| `cb_reserves_total` | Reserves totales banques centrales |
| `cb_net_purchases_monthly` | Achats nets du mois |
| `cb_purchase_trend_3m` | Tendance achats 3 mois |
| `cb_purchase_trend_12m` | Tendance achats 12 mois |
| `china_gold_reserves` | Reserves or Chine |
| `russia_gold_reserves` | Reserves or Russie |
| `cb_buying_regime` | 1 si achats nets positifs |

### 22. Options & Derivatives (CME, CBOE)

**Source: Yahoo (GVZ gratuit), Barchart (limite gratuit)**

| Feature | Description | Source |
|---------|-------------|--------|
| `gvz` | Gold VIX (implied volatility) | ^GVZ Yahoo |
| `gvz_percentile` | Percentile GVZ sur 1 an | Calcule |
| `gvz_change` | Changement GVZ | Calcule |
| `gvz_vs_realized` | GVZ vs volatilite realisee | Calcule |
| `gld_put_call_ratio` | Put/Call ratio GLD | Barchart |
| `gld_options_volume` | Volume options GLD | Yahoo |
| `implied_vs_realized_vol` | Ratio IV/RV | Calcule |
| `vol_risk_premium` | Prime de risque vol | Calcule |

---

## 27. GAMMA EXPOSURE (GEX) - MARKET MICROSTRUCTURE

**Le Gamma Exposure revele comment les Market Makers vont AMPLIFIER ou STABILISER les mouvements de prix.**

### Concept

```
GEX POSITIF (Dealers Long Gamma):
  - Prix monte → Dealers VENDENT pour hedger → RESISTANCE
  - Prix baisse → Dealers ACHETENT pour hedger → SUPPORT
  → Marche STABLE, faible volatilite, RANGE

GEX NEGATIF (Dealers Short Gamma):
  - Prix monte → Dealers ACHETENT pour hedger → AMPLIFIE la hausse
  - Prix baisse → Dealers VENDENT pour hedger → AMPLIFIE la baisse
  → Marche VOLATILE, mouvements EXPLOSIFS
```

### Sources GRATUITES pour GEX

| Source | Donnees | Acces | Limite |
|--------|---------|-------|--------|
| **Barchart.com** | GEX GLD, GEX IAU | Gratuit | 5 requetes/jour |
| **CBOE** | Options Open Interest | Gratuit | Daily EOD |
| **CME Group** | Gold Options OI | Gratuit | Daily EOD |
| **Yahoo Finance** | Options chains GLD | Gratuit | 15min delay |
| **GitHub gex-tracker** | Script Python | Open source | Illimite |

### Features GEX

| Feature | Description | Source |
|---------|-------------|--------|
| `gex_gld` | Gamma Exposure GLD | Barchart/Calcule |
| `gex_gld_normalized` | GEX / moyenne 30j | Calcule |
| `gex_regime` | 1 si GEX > 0, -1 si < 0 | Calcule |
| `gamma_flip_level` | Prix ou GEX passe de + a - | Calcule |
| `above_gamma_flip` | 1 si prix > gamma flip | Calcule |
| `call_wall` | Strike avec max OI calls | CME/Yahoo |
| `put_wall` | Strike avec max OI puts | CME/Yahoo |
| `distance_to_call_wall` | (Call Wall - Prix) / Prix | Calcule |
| `distance_to_put_wall` | (Prix - Put Wall) / Prix | Calcule |
| `max_pain` | Strike ou options expirent sans valeur | Calcule |
| `distance_to_max_pain` | Distance au max pain | Calcule |
| `gex_change_1d` | Changement GEX 1 jour | Calcule |
| `gex_momentum` | Tendance GEX 5 jours | Calcule |
| `put_call_oi_ratio` | Put OI / Call OI | CME/Yahoo |
| `options_volume_ratio` | Volume Calls / Puts | CME/Yahoo |

### Interpretation Trading

| Regime GEX | Strategie | Taille Position |
|------------|-----------|-----------------|
| GEX > 0 (Positif) | Mean-reversion, fade les breakouts | Reduire (0.33%) |
| GEX < 0 (Negatif) | Trend-following, breakouts | Normale (0.66-1%) |
| Prix > Gamma Flip + GEX < 0 | Breakout haussier probable | Full (1%) |
| Prix < Gamma Flip + GEX < 0 | Breakout baissier probable | Full (1%) |

### Code Python - Calculer GEX (GRATUIT)

```python
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_gld_options_data():
    """Recupere les options GLD depuis Yahoo Finance (GRATUIT)"""
    gld = yf.Ticker("GLD")
    
    # Prix spot
    spot_price = gld.info.get('regularMarketPrice', gld.history(period='1d')['Close'].iloc[-1])
    
    # Dates d'expiration
    expirations = gld.options[:4]  # 4 prochaines expirations
    
    all_options = []
    for exp in expirations:
        opt = gld.option_chain(exp)
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        calls['type'] = 'call'
        puts['type'] = 'put'
        calls['expiration'] = exp
        puts['expiration'] = exp
        all_options.extend([calls, puts])
    
    return pd.concat(all_options), spot_price

def calculate_gex(options_df, spot_price):
    """
    Calcule le Gamma Exposure (GEX)
    
    Formule: GEX = Gamma * Open Interest * Spot Price * 100 * (-1 si put)
    
    Hypothese: Dealers sont LONG calls (clients achetent calls)
               Dealers sont SHORT puts (clients achetent puts)
    """
    options_df = options_df.copy()
    
    # Gamma * OI * Spot * 100 (multiplicateur options)
    options_df['gex_contribution'] = (
        options_df['gamma'] * 
        options_df['openInterest'] * 
        spot_price * 
        100
    )
    
    # Inverser le signe pour les puts (dealers short puts = short gamma)
    options_df.loc[options_df['type'] == 'put', 'gex_contribution'] *= -1
    
    # GEX total
    total_gex = options_df['gex_contribution'].sum()
    
    # GEX par strike
    gex_by_strike = options_df.groupby('strike')['gex_contribution'].sum()
    
    return total_gex, gex_by_strike

def find_key_levels(gex_by_strike, spot_price):
    """Trouve les niveaux cles: Call Wall, Put Wall, Gamma Flip"""
    
    # Call Wall = Strike avec le plus gros GEX positif (resistance)
    call_wall = gex_by_strike[gex_by_strike > 0].idxmax() if (gex_by_strike > 0).any() else None
    
    # Put Wall = Strike avec le plus gros GEX negatif (support)
    put_wall = gex_by_strike[gex_by_strike < 0].idxmin() if (gex_by_strike < 0).any() else None
    
    # Gamma Flip = Prix ou GEX cumule passe de + a -
    gex_cumsum = gex_by_strike.sort_index().cumsum()
    flip_candidates = gex_cumsum[gex_cumsum.shift(1) * gex_cumsum < 0]
    gamma_flip = flip_candidates.index[0] if len(flip_candidates) > 0 else spot_price
    
    return {
        'call_wall': call_wall,
        'put_wall': put_wall,
        'gamma_flip': gamma_flip,
        'above_gamma_flip': spot_price > gamma_flip
    }

def get_gex_features():
    """Feature engineering complet pour GEX"""
    options_df, spot = get_gld_options_data()
    total_gex, gex_by_strike = calculate_gex(options_df, spot)
    levels = find_key_levels(gex_by_strike, spot)
    
    features = {
        'gex_total': total_gex,
        'gex_normalized': total_gex / 1e9,  # En milliards
        'gex_regime': 1 if total_gex > 0 else -1,
        'call_wall': levels['call_wall'],
        'put_wall': levels['put_wall'],
        'gamma_flip': levels['gamma_flip'],
        'above_gamma_flip': int(levels['above_gamma_flip']),
        'distance_to_call_wall': (levels['call_wall'] - spot) / spot if levels['call_wall'] else 0,
        'distance_to_put_wall': (spot - levels['put_wall']) / spot if levels['put_wall'] else 0,
    }
    
    return features

# Exemple d'utilisation
if __name__ == "__main__":
    features = get_gex_features()
    print("=== GEX Features ===")
    for k, v in features.items():
        print(f"{k}: {v}")
```

### Alternative: Scraper Barchart (Plus precis)

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_barchart_gex(symbol="GLD"):
    """
    Scrape GEX depuis Barchart.com (GRATUIT, limite 5/jour)
    URL: https://www.barchart.com/etfs-funds/quotes/GLD/gamma-exposure
    """
    url = f"https://www.barchart.com/etfs-funds/quotes/{symbol}/gamma-exposure"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parser les donnees GEX (structure peut changer)
    # Retourne les niveaux de gamma par strike
    
    return soup  # A adapter selon la structure HTML

# Note: Pour usage intensif, utiliser le calcul Yahoo Finance ci-dessus
```

### Combo GEX + ETF Flows = Setups Optimaux

```python
def get_optimal_setup(gex_regime, etf_flow_direction, price_vs_gamma_flip):
    """
    Combine GEX et ETF Flows pour determiner le setup optimal
    """
    
    # SETUP 1: Explosion Haussiere
    if gex_regime == -1 and etf_flow_direction == 1 and price_vs_gamma_flip == 1:
        return {
            'signal': 'STRONG_LONG',
            'size': 1.0,  # Full size
            'reason': 'GEX negatif + ETF inflows + Prix > Gamma Flip'
        }
    
    # SETUP 2: Explosion Baissiere
    if gex_regime == -1 and etf_flow_direction == -1 and price_vs_gamma_flip == -1:
        return {
            'signal': 'STRONG_SHORT',
            'size': 1.0,
            'reason': 'GEX negatif + ETF outflows + Prix < Gamma Flip'
        }
    
    # SETUP 3: Range (pas de trade)
    if gex_regime == 1:
        return {
            'signal': 'NO_TRADE',
            'size': 0.0,
            'reason': 'GEX positif = Market makers stabilisent le prix'
        }
    
    # SETUP 4: Signal faible
    return {
        'signal': 'WEAK',
        'size': 0.33,
        'reason': 'Signaux mixtes'
    }
```

---

### 23. Futures Market Structure (CME)

**Source: Yahoo Finance (Gratuit)**

| Feature | Description | Ticker |
|---------|-------------|--------|
| `gold_futures_front` | Prix futures front month | GC=F |
| `gold_futures_basis` | Futures - Spot | Calcule |
| `gold_contango_pct` | % contango/backwardation | Calcule |
| `futures_roll_yield` | Yield du roll | Calcule |
| `futures_open_interest` | Open Interest | Yahoo |
| `futures_oi_change` | Changement OI | Calcule |
| `futures_volume` | Volume futures | Yahoo |
| `term_structure_slope` | Pente term structure | Calcule |

### 24. Geopolitical & Risk Sentiment

**Source: Divers (Gratuit)**

| Feature | Description | Source |
|---------|-------------|--------|
| `gpr_index` | Geopolitical Risk Index | policyuncertainty.com |
| `gpr_change` | Changement GPR | Calcule |
| `vix` | VIX (fear index) | ^VIX Yahoo |
| `vix_term_structure` | VIX vs VIX3M | Yahoo |
| `credit_spread_hy` | High Yield spread | FRED BAMLH0A0HYM2 |
| `risk_off_signal` | Signal risk-off agrege | Calcule |
| `safe_haven_demand` | Proxy demande safe haven | Calcule |
| `stock_bond_correlation` | Correl SPY/TLT rolling | Calcule |

### 25. Dollar & Yields

**Source: Yahoo Finance (Gratuit)**

| Feature | Description | Ticker |
|---------|-------------|--------|
| `dxy` | Dollar Index | DX-Y.NYB |
| `dxy_change_5d` | Changement DXY 5j | Calcule |
| `dxy_trend` | Tendance DXY 20j | Calcule |
| `us10y_yield` | US 10Y Treasury yield | ^TNX |
| `us2y_yield` | US 2Y Treasury yield | ^IRX |
| `yield_curve_slope` | 10Y - 2Y (inversion) | Calcule |
| `us10y_change` | Changement 10Y | Calcule |
| `dollar_gold_divergence` | Divergence DXY/Gold | Calcule |

### 26. Cross-Asset Correlations (Two Sigma style)

**Source: Calcule a partir de Yahoo**

| Feature | Description |
|---------|-------------|
| `gold_spy_corr_20d` | Correlation Gold/SPY 20j |
| `gold_tlt_corr_20d` | Correlation Gold/Bonds 20j |
| `gold_dxy_corr_20d` | Correlation Gold/Dollar 20j |
| `gold_oil_corr_20d` | Correlation Gold/Oil 20j |
| `stock_bond_corr_20d` | Correlation SPY/TLT 20j |
| `correlation_regime` | Regime de correlation |
| `diversification_value` | Valeur diversification Gold |

---

## Categories de Features Techniques (par TF)

### 1-16. (Inchange - voir sections precedentes)

- Price Action, Patterns Candlestick, Patterns Chart
- Moving Averages, Momentum, MACD, Volatilite
- Volume, Trend Strength, Ichimoku, Autres
- Cross-Asset (DXY, VIX, XAGUSD, USDJPY, USDCHF)
- COT Data, Saisonnalites, Temporel, Calendrier

---

## Sources de Donnees - Resume

### Gratuites et Live

| Donnee | Source | API/Package | Frequence |
|--------|--------|-------------|-----------|
| XAUUSD | MetaTrader 5 | MetaTrader5 | Tick |
| Cross-assets (XAGUSD, USDJPY, USDCHF) | MetaTrader 5 | MetaTrader5 | Tick |
| DXY, VIX, GVZ, US10Y | Yahoo Finance | yfinance | 15min delay |
| TIPS, Fed Rate, Breakevens | FRED | fredapi | Daily |
| COT Data | CFTC | cot-reports | Weekly |
| GLD Holdings | SPDR Website | Web scraping | Daily |
| GLD/IAU Options (pour GEX) | Yahoo Finance | yfinance | 15min delay |
| GEX pre-calcule | Barchart.com | Web scraping | Daily (limite 5/jour) |
| GPR Index | policyuncertainty.com | Download | Monthly |
| Central Bank Reserves | World Gold Council | Download | Monthly |

### Installation des packages

```bash
pip install MetaTrader5 yfinance fredapi pandas-datareader
pip install cot-reports requests beautifulsoup4
pip install scipy  # Pour calculs gamma
```

### Exemple de code pour FRED

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_FREE_API_KEY')

# Real rates (TIPS)
tips_10y = fred.get_series('DFII10')

# Breakeven inflation
breakeven_10y = fred.get_series('T10YIE')

# Fed Funds Rate
fed_rate = fred.get_series('FEDFUNDS')
```

### Exemple pour GLD Holdings

```python
import yfinance as yf

# GLD ETF data
gld = yf.Ticker("GLD")
gld_history = gld.history(period="5y")

# Pour les holdings exacts, scraper SPDR
# https://www.spdrgoldshares.com/usa/historical-data/
```

### Exemple pour GEX (voir section 27)

```python
# Voir la section 27 pour le code complet de calcul GEX
from data.gamma_loader import get_gex_features

gex_features = get_gex_features()
print(f"GEX Regime: {'Stable' if gex_features['gex_regime'] > 0 else 'Volatile'}")
print(f"Call Wall: {gex_features['call_wall']}")
print(f"Put Wall: {gex_features['put_wall']}")
```

---

## Lags (Option B)

Pour les indicateurs cles :
- Lag 1, 5, 10 periodes
- Change sur 5 et 10 periodes
- Slope (tendance)
- Acceleration (changement de slope)

---

## Reward Function (Professionnel)

Base sur les standards academiques :
- Differential Sharpe Ratio (Moody & Saffell, 1998)
- Risk-adjusted PnL
- Multi-objective optimization

### Composantes

| Component | Poids | Source |
|-----------|-------|--------|
| Differential Sharpe | 0.4 | Performance risk-adjusted |
| Risk-adjusted PnL | 0.3 | Recompense bons trades |
| Drawdown Penalty | 1.0 | Evite gros DD |
| Turnover Cost | 1.0 | Anti-overtrading |
| Consecutive Loss Penalty | 1.0 | Protection capitale |
| BE Bonus | 1.0 | Securise les gains |

**Les poids seront optimises avec Optuna.**

---

## Regles FTMO

| Regle | Limite | Implementation |
|-------|--------|----------------|
| Max Drawdown | 10% | Episode termine + reward -100 |
| Daily Drawdown | 5% | Episode termine + reward -100 |
| Profit Target | 10% | Bonus reward +50 |
| Min Trading Days | 4 jours | Penalite si < 4 trades |

---

## Training

### Configuration

| Element | Valeur |
|---------|--------|
| Episodes | Long (3 mois par episode) |
| Validation | Walk-Forward |
| Optimisation | Optuna (hyperparametres + reward weights) |
| Tracking | Weights & Biases (tradingluca31) |

### Simulation

| Element | Valeur |
|---------|--------|
| Spread | 20 points |
| Slippage | 5 points |
| Capital initial | 10,000 USD |

---

## Structure du Projet

```
Pikachu-/
|
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
|
├── config/
│   ├── settings.py
│   ├── ftmo_rules.py
│   └── hyperparameters.yaml
|
├── data/
│   ├── data_loader.py
│   ├── mt5_loader.py           # MetaTrader 5 data
│   ├── fred_loader.py          # FRED macro data
│   ├── yahoo_loader.py         # Yahoo cross-assets
│   ├── cot_loader.py           # COT data CFTC
│   ├── etf_flows_loader.py     # GLD holdings/flows
│   ├── gamma_loader.py         # GEX calculation (NEW)
│   ├── feature_engineer.py
│   ├── preprocessor.py
│   └── validators.py
|
├── environment/
│   ├── trading_env.py
│   ├── reward_function.py
│   ├── position_sizing.py
│   └── sl_tp_manager.py
|
├── agents/
│   ├── sac_agent.py
│   └── callbacks.py
|
├── training/
│   ├── trainer.py
│   ├── optuna_optimizer.py
│   └── walk_forward.py
|
├── evaluation/
│   ├── backtester.py
│   ├── metrics.py
│   └── ftmo_validator.py
|
├── live/
│   ├── mt5_connector.py
│   ├── data_stream.py          # Real-time data aggregation
│   └── live_trader.py
|
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_macro_features.ipynb  # Analyse features macro
│   ├── 04_gamma_analysis.ipynb  # Analyse GEX (NEW)
│   ├── 05_training_analysis.ipynb
│   └── 06_backtest_results.ipynb
|
├── tests/
│   └── ...
|
└── scripts/
    ├── download_all_data.py     # Telecharge toutes les donnees
    ├── update_macro_data.py     # Update donnees macro
    ├── update_gamma_data.py     # Update GEX daily (NEW)
    ├── train.py
    ├── optimize.py
    ├── backtest.py
    └── live.py
```

---

## Phases du Projet

### Phase 1 : Data Pipeline
- [ ] Telecharger donnees XAUUSD (MT5)
- [ ] Telecharger donnees cross-asset (Yahoo + MT5)
- [ ] Telecharger donnees macro (FRED - TIPS, Fed, Inflation)
- [ ] Telecharger donnees COT (CFTC)
- [ ] Telecharger GLD holdings/flows
- [ ] Calculer GEX depuis options GLD (NEW)
- [ ] Feature engineering (~1200-1800 features)
- [ ] Validation anti-leak, anti-lookahead
- [ ] Split train/val/test

### Phase 2 : Environment
- [ ] Creer Gymnasium environment
- [ ] Implementer reward function
- [ ] Implementer position sizing
- [ ] Implementer SL/TP + BE
- [ ] Tests unitaires

### Phase 3 : Training
- [ ] Configurer SAC
- [ ] Training baseline
- [ ] Optuna optimization
- [ ] Walk-forward validation
- [ ] Tracking W&B

### Phase 4 : Evaluation
- [ ] Backtest sur test set (2024-2026)
- [ ] Metriques (Sharpe, Sortino, Max DD, Win Rate)
- [ ] Validation regles FTMO
- [ ] Analyse des trades
- [ ] Feature importance analysis
- [ ] Analyse impact GEX/ETF Flows (NEW)

### Phase 5 : Paper Trading
- [ ] Connexion MT5
- [ ] Real-time data streaming (MT5 + Yahoo + FRED + GEX)
- [ ] Paper trading sur compte demo
- [ ] Validation en conditions reelles

### Phase 6 : FTMO Challenge
- [ ] Challenge sur compte reel
- [ ] Monitoring continu
- [ ] Payout tous les 15 jours

---

## Technologies

- **Python 3.10+**
- **Stable-Baselines3** : SAC implementation
- **Gymnasium** : Environment
- **PyTorch** : Neural networks
- **Pandas/NumPy** : Data processing
- **TA-Lib** : Indicateurs techniques
- **MetaTrader5** : Broker + XAUUSD, XAGUSD, USDJPY, USDCHF
- **yfinance** : DXY, VIX, GVZ, GLD, US10Y, Options chains
- **fredapi** : TIPS, Fed Rate, Breakevens, Balance Sheet
- **cot-reports** : COT data CFTC
- **BeautifulSoup** : Web scraping (Barchart GEX)
- **scipy** : Calculs gamma/options
- **Optuna** : Hyperparameter optimization
- **Weights & Biases** : Experiment tracking
- **pytest** : Testing

---

## API Keys Requises (Gratuites)

| Service | Comment obtenir |
|---------|-----------------|
| FRED | https://fred.stlouisfed.org/docs/api/api_key.html |
| MT5 | Compte demo chez broker (FTMO, etc.) |
| Yahoo | Pas de cle requise |
| Barchart | Pas de cle requise (limite 5 requetes/jour) |
| W&B | https://wandb.ai/authorize |

---

## Auteur

- **Luca** - Trading algorithmique & Reinforcement Learning
- **Objectif** : Passer FTMO avec un agent RL

---

## License

MIT License
