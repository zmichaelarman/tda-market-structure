# tda-market-structure

> Applying persistent homology to equity return data to detect topological signatures of market stress regimes.

This project applies persistent homology to multi-dimensional equity return data to see if topological signatures can detect or predict market stress regimes.

The goal here is to bridge the gap between abstract algebraic topology research and practical quantitative finance by building a pipeline that extracts Betti number time series from S&P 500 returns and tests their true predictive power over market drawdowns.

---

## Visualizations

<img width="1823" height="1248" alt="SP500_return_correlation_map" src="https://github.com/user-attachments/assets/47b9707a-135e-4a11-8d8b-3bd6a733be3f" />

<img width="1957" height="1579" alt="Calm_vs_Crisis_return_cloud" src="https://github.com/user-attachments/assets/a1a20fdc-296e-4adc-afbf-105954c2b614" />

---

## Background

Most quantitative signals lean on standard statistical summaries like rolling volatilities, correlation matrices, or momentum factors. This project takes a different angle. Does the geometric shape of the return distribution change before the market breaks?

By treating a rolling window of daily stock returns as a multi-dimensional point cloud, we can use topological data analysis (TDA) to track how the market's underlying geometry (aka: clusters, loops, and voids) evolves over time.

This project was inspired from my research conducted at the [Mason Experimental Geometry Lab](https://megl.science.gmu.edu/), where similar mathematical machinery was applied to spin systems and percolation models in [ATEAMS](https://github.com/apizzimenti/ATEAMS). This project translates those core mathematical concepts from statistical physics to financial markets & time series.

---

## Research Question

> Do Betti numbers and persistence statistics derived from S&P 500 return point clouds carry statistically significant predictive signals over volatility regimes or market drawdowns/stress events?

Analysis focuses on five historical stress periods as validation anchors:

| Event | Period |
|---|---|
| Global Financial Crisis | Sep 2008 – Jun 2009 |
| Flash Crash | May 2010 – Jun 2010 |
| Q4 Drawdown | Oct 2018 – Dec 2018 |
| COVID-19 Crash | Feb 2020 – Apr 2020 |
| Fed Rate Hike Cycle | Jan 2022 – Oct 2022 |

---

## Mathematical Approach

**Point cloud construction.** For each trading day $t$, we take a rolling window of the previous $W = 30$ days of log returns across $N$ stocks. Each day becomes a point in $\mathbb{R}^N$, giving a point cloud $X_t \subset \mathbb{R}^N$ of 30 points. Returns are standardized per-stock within each window, and PCA reduces the ambient dimension to 10 components before computing persistence.

**Persistent homology.** I compute the Vietoris-Rips filtration on $X_t$ using [Ripser](https://github.com/scikit-tda/ripser.py). The output is a persistence diagram $\text{PD}(X_t)$ encoding the birth and death of topological features across scales.

---

## Results
the topological features behave as acoincident indicator of
market stress, not a leading one. They carry statistically 
significant information about the *current* state of the market, 
but show no reliable power to forecast forward drawdowns.

### Key Findings

* **Contemporaneous structure is strong.** Spearman rank correlations against
  same-day volatility are large and highly significant: H0 total persistence vs.
  21-day realized volatility reaches ρ = −0.58 (ρ = −0.41 vs. the VIX), with β₁ and
  H1 entropy near ρ = −0.17. Lower H0 persistence corresponds to a more tightly
  clustered point cloud and cross-sectional correlations rising as stocks move together.

* **Forward prediction is weak to absent.** Against the actual stress target every correlation is
  negligible (|ρ| < 0.06 at all horizons), even where it clears FDR purely on sample size (n ≈ 5,200). 
  Correlations with forward volatility are larger (ρ ≈ −0.22 for H0 persistence) but are driven by the
  same H0/dispersion term acting as a volatility proxy under volatility clustering.

* **Stress vs. normal distributions.** Mann-Whitney U tests find three of five features
  shift significantly between stress and non-stress periods after FDR, H0 total
  persistence, β₁, and H1 entropy (rank-biserial r = 0.34, 0.16, 0.15) are all lower
  during stress. The two purely loop-magnitude features (H1 total / max persistence)
  do not separate the regimes.

* **Granger causality.** After correcting the forward-volatility target, all five
  features Granger-predict forward 21-day realized volatility at lag 1 (FDR p < 0.03),
  but none Granger-predict the VIX.

* **Out-of-sample backtest.** A z-score timing rule on the strongest feature
  (H0 total persistence) matches buy-and-hold on held-out data (test Sharpe 0.64 vs.
  0.65; 11.4% vs. 11.6% annualized) and is marginally worse after 2 bps costs.
---

## Discussion

The pipeline show thatpersistent homology of S&P 500 return point 
clouds detects market stress as ithappens but does not anticipate
it. Three points explain the pattern.

---

## Conclusion

The results show that persistent homology features of S&P 500 return point clouds track market
stress contemporaneously but provide no exploitable predictive signal over forward
drawdowns across the period 2004–2024.

---

## Installation

**Requirements:** Python 3.9+

```bash
git clone https://github.com/zmichaelarman/tda-market-structure.git
cd tda-market-structure
python -m venv env
source env/bin/activate       # Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

Run the pipeline in order from the project root:

```bash
#Download and clean 20 years of S&P 500 return data
python data/fetch.py

#Compute persistence diagrams (~10 min for full history)
python topology/pipeline.py

#Extract topological feature time series
python features/extract.py

#Visual validation against known stress events
python analysis/validate.py

#Statistical analysis (results written to data/regime_report.txt)
python analysis/regimes.py

#Backtest (run only if significant signal found in regimes.py)
python backtest/backtest.py
```

All scripts are self-contained and write their outputs to `data/` and `figures/`.
Each script logs progress and expected runtime to the terminal.

---

## Dependencies
| Library | Version |
|---|---|
| `yfinance` | 1.2.0 |
| `ripser` | 0.6.15 |
| `persim` | 0.3.8 |
| `scikit-learn` | 1.1 |
| `pandas` | 1.5 |
| `numpy` | 1.23 |
| `scipy` | 1.9 |
| `statsmodels` | 0.13 |
| `matplotlib` | 3.6 |

---

## Related Work

This project is inspired by my work on the repository [ATEAMS](https://github.com/apizzimenti/ATEAMS). I haved used what I learned and attempted to apply it to the financial markets.

---

## References

Bauer, U. (2021). Ripser: Efficient computation of Vietoris–Rips persistence barcodes. *Journal of Applied and Computational Topology*, 5(3), 391–423. https://doi.org/10.1007/s41468-021-00071-5

Bouchaud, J.-P., & Potters, M. (2003). *Theory of Financial Risk and Derivative Pricing* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511753893

Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255–308. https://doi.org/10.1090/S0273-0979-09-01249-X

Edelsbrunner, H., Letscher, D., & Zomorodian, A. (2002). Topological persistence and simplification. *Discrete & Computational Geometry*, 28(4), 511–533. https://doi.org/10.1007/s00454-002-2885-2

Engle, R. (2002). Dynamic conditional correlation. *Journal of Business & Economic Statistics*, 20(3), 339–350. https://doi.org/10.1198/073500102288618487

Gidea, M. (2017). Topological data analysis of critical transitions in financial networks. In *NetSci-X 2017*, Springer Proceedings in Complexity (pp. 47–59). https://doi.org/10.1007/978-3-319-55471-6_5

Gidea, M., & Katz, Y. (2018). Topological data analysis of financial time series: Landscapes of crashes. *Physica A*, 491, 820–834. https://doi.org/10.1016/j.physa.2017.09.028

Ismail, M. S., et al. (2022). Early warning signals of financial crises using persistent homology. *Physica A*, 586, 126459. https://doi.org/10.1016/j.physa.2021.126459

Mantegna, R. N. (1999). Hierarchical structure in financial markets. *European Physical Journal B*, 11(1), 193–197. https://doi.org/10.1007/s100510050929

Tralie, C., Saul, N., & Bar-On, R. (2018). Ripser.py: A lean persistent homology library for Python. *Journal of Open Source Software*, 3(29), 925. https://doi.org/10.21105/joss.00925

Zomorodian, A., & Carlsson, G. (2005). Computing persistent homology. *Discrete & Computational Geometry*, 33(2), 249–274. https://doi.org/10.1007/s00454-004-1146-y

---

## Author

**Michael Arman**
Mathematics, George Mason University (May 2026)

[LinkedIn](https://linkedin.com/in/zmichaelarman) · [GitHub](https://github.com/zmichaelarman) · zmichaelarman@gmail.com

---

## Citation

```bibtex
@software{arman2026tda,
  title  = {tda-market-structure: Persistent Homology for Equity Market Analysis},
  author = {Arman, Michael},
  year   = {2026},
  url    = {https://github.com/zmichaelarman/tda-market-structure}
}
```

---
