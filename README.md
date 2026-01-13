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
| Sources | MetaTrader 5 + Yahoo Finance + CFTC (COT) |
| Timeframes | D1, H4, H1, M15 (Multi-TF) |
| Split | Train 70% (2015-2022) / Val 15% (2022-2024) / Test 15% (2024-2026) |

### Algorithme

| Element | Valeur |
|---------|--------|
| Algorithme | SAC (Soft Actor-Critic) |
| Framework | Stable-Baselines3 |
| Action Space | Continu [-1, +1] discretise en 3 niveaux |
| Observation | ~800-1000 features (Option B: features + lags) |

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

## Features (~800-1000)

### Architecture Multi-Timeframe

```
D1  --> Tendance majeure + contexte
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

### Categories de Features (par TF)

#### 1. Price Action
- Returns (1, 5, 20 periodes)
- Log returns
- High-Low range
- Close position dans la range

#### 2. Patterns Candlestick
- Doji, Hammer, Inverted Hammer
- Engulfing (bullish/bearish)
- Morning Star, Evening Star
- Three White Soldiers, Three Black Crows
- Harami, Piercing Line, Dark Cloud Cover
- Tweezer Top/Bottom
- Et autres...

#### 3. Patterns Chart
- Double Top/Bottom
- Head & Shoulders
- Triangles (ascending, descending, symmetrical)
- Flags (bull/bear)
- Wedges (rising/falling)

#### 4. Moving Averages
- EMA 8, 13, 20, 55, 200
- SMA 50, 200
- SMMA 50, 200
- WMA 20
- Cross signals (EMA 20/55, SMMA 50/200)

#### 5. Momentum
- RSI (7, 14, 21)
- Stochastic (%K, %D)
- Stochastic RSI
- CCI
- Williams %R
- ROC
- Momentum

#### 6. MACD
- MACD Line
- Signal Line
- Histogram

#### 7. Volatilite
- ATR (7, 14, 21)
- Bollinger Bands (upper, middle, lower, width, %B)
- Keltner Channels
- Donchian Channels

#### 8. Volume
- Volume SMA
- Volume Ratio
- OBV
- VWAP + distance
- MFI
- A/D Line
- CMF

#### 9. Trend Strength
- ADX
- +DI / -DI
- ADX trend signal

#### 10. Ichimoku
- Tenkan-sen
- Kijun-sen
- Senkou Span A/B
- Chikou Span
- Price vs Cloud

#### 11. Autres Indicateurs
- Pivot Points (R1, S1)
- Parabolic SAR
- Supertrend

#### 12. Cross-Asset (Global) - Correlations

| Asset | Correlation avec Gold | Source |
|-------|----------------------|--------|
| DXY | Inverse (-0.8) | Yahoo |
| US10Y | Inverse | Yahoo |
| VIX | Positive (risk-off) | Yahoo |
| XAGUSD | Positive (+0.9) | MT5 |
| USDJPY | Inverse (USD strength) | MT5 |
| USDCHF | Inverse (USD strength) | MT5 |

Features par cross-asset:
- Prix normalise
- Returns
- Distance a la moyenne
- Correlation rolling avec Gold

#### 13. COT (Commitment of Traders) - CFTC Data

Donnees hebdomadaires du rapport COT sur Gold Futures:

| Feature | Description |
|---------|-------------|
| `cot_commercial_long` | Positions long des commerciaux (hedgers) |
| `cot_commercial_short` | Positions short des commerciaux |
| `cot_commercial_net` | Net position commerciaux |
| `cot_noncommercial_long` | Positions long des speculateurs |
| `cot_noncommercial_short` | Positions short des speculateurs |
| `cot_noncommercial_net` | Net position speculateurs (IMPORTANT) |
| `cot_open_interest` | Open interest total |
| `cot_change_noncommercial` | Changement hebdo net speculateurs |
| `cot_extreme_long` | 1 si positioning extreme long |
| `cot_extreme_short` | 1 si positioning extreme short |

**Interpretation:**
- Speculateurs net long extreme = potentiel top (contrarian)
- Speculateurs net short extreme = potentiel bottom (contrarian)
- Changement important = momentum institutionnel

#### 14. Saisonnalites (Seasonality)

| Feature | Description |
|---------|-------------|
| `month` | Mois (1-12) encode sin/cos |
| `day_of_month` | Jour du mois (1-31) |
| `week_of_year` | Semaine de l'annee (1-52) |
| `is_month_end` | 1 si derniers 3 jours du mois |
| `is_month_start` | 1 si premiers 3 jours du mois |
| `is_quarter_end` | 1 si fin de trimestre |
| `is_year_end` | 1 si decembre |
| `gold_seasonal_bullish` | Mois historiquement bullish (Jan, Fev, Aout, Sept) |
| `gold_seasonal_bearish` | Mois historiquement bearish (Mars, Juin) |
| `is_options_expiry` | 1 si semaine d'expiration options |
| `is_futures_rollover` | 1 si periode de rollover |
| `days_to_nfp` | Jours avant prochain NFP |
| `days_to_fomc` | Jours avant prochain FOMC |

**Patterns saisonniers Gold connus:**
- Janvier-Fevrier: Souvent bullish (demande bijoux Asie)
- Aout-Septembre: Souvent bullish (saison mariages Inde)
- Mars: Souvent correction
- Fin de mois: Rebalancing institutionnel

#### 15. Temporel (Global)

- Hour (sin/cos encoded)
- Day of week
- Is London session
- Is NY session
- Is London/NY overlap

#### 16. Calendrier Economique (Global)

- News impact (1h, 4h)
- Is NFP day
- Is FOMC day
- Minutes to next high impact news

### Lags (Option B)

Pour les indicateurs cles :
- Lag 1, 5, 10 periodes
- Change sur 5 et 10 periodes
- Slope (tendance)

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

### Penalites Drawdown

```
0%  -------- Safe Zone
3%  -------- Warning (-2x penalty)
5%  -------- Danger Daily DD
6%  -------- Serious (-5x penalty)
8%  -------- Critical (-20x penalty)
10% -------- FAIL (episode termine)
```

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

### Walk-Forward Validation

```
Train: 2015-2018 --> Test: 2018-2019
Train: 2015-2019 --> Test: 2019-2020
Train: 2015-2020 --> Test: 2020-2021
...
```

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
│   ├── feature_engineer.py
│   ├── preprocessor.py
│   ├── cot_loader.py          # COT data from CFTC
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
│   └── live_trader.py
|
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_training_analysis.ipynb
│   └── 04_backtest_results.ipynb
|
├── tests/
│   ├── test_env.py
│   ├── test_features.py
│   └── test_reward.py
|
└── scripts/
    ├── download_data.py
    ├── download_cot.py         # Script COT CFTC
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
- [ ] Telecharger donnees COT (CFTC)
- [ ] Feature engineering (~800-1000 features)
- [ ] Ajouter saisonnalites
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

### Phase 5 : Paper Trading
- [ ] Connexion MT5
- [ ] Paper trading sur compte demo
- [ ] Validation en conditions reelles
- [ ] Ajustements si necessaire

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
- **MetaTrader5** : Broker connection + XAGUSD, USDJPY, USDCHF
- **yfinance** : Cross-asset data (DXY, VIX, US10Y)
- **cot-reports** : COT data from CFTC
- **Optuna** : Hyperparameter optimization
- **Weights & Biases** : Experiment tracking
- **pytest** : Testing

---

## Auteur

- **Luca** - Trading algorithmique & Reinforcement Learning
- **Objectif** : Passer FTMO avec un agent RL

---

## License

MIT License
