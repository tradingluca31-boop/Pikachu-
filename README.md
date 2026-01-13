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
| Sources | MetaTrader 5 + Yahoo + FRED + CFTC |
| Timeframes | D1, H4, H1, M15 (Multi-TF) |
| Split | Train 70% (2015-2022) / Val 15% (2022-2024) / Test 15% (2024-2026) |

### Algorithme

| Element | Valeur |
|---------|--------|
| Algorithme | SAC (Soft Actor-Critic) |
| Framework | Stable-Baselines3 |
| Action Space | Continu [-1, +1] discretise en 3 niveaux |
| Observation | ~1000-1500 features (Option B: features + lags) |

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

## Features (~1000-1500)

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

### 20. Gold ETF Flows (World Gold Council)

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

**Interpretation:**
- Flows positifs = Demande institutionnelle
- Flows negatifs = Distribution
- Acceleration = Changement de sentiment

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
| `gamma_exposure_approx` | GEX approxime (si dispo) | Calcule |

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
| GPR Index | policyuncertainty.com | Download | Monthly |
| Central Bank Reserves | World Gold Council | Download | Monthly |

### Installation des packages

```bash
pip install MetaTrader5 yfinance fredapi pandas-datareader
pip install cot-reports requests beautifulsoup4
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
│   ├── 04_training_analysis.ipynb
│   └── 05_backtest_results.ipynb
|
├── tests/
│   └── ...
|
└── scripts/
    ├── download_all_data.py     # Telecharge toutes les donnees
    ├── update_macro_data.py     # Update donnees macro
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
- [ ] Feature engineering (~1000-1500 features)
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

### Phase 5 : Paper Trading
- [ ] Connexion MT5
- [ ] Real-time data streaming (MT5 + Yahoo + FRED)
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
- **yfinance** : DXY, VIX, GVZ, GLD, US10Y
- **fredapi** : TIPS, Fed Rate, Breakevens, Balance Sheet
- **cot-reports** : COT data CFTC
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
| W&B | https://wandb.ai/authorize |

---

## Auteur

- **Luca** - Trading algorithmique & Reinforcement Learning
- **Objectif** : Passer FTMO avec un agent RL

---

## License

MIT License
