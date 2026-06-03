# tda-market-structure

> Applying persistent homology to equity return data to detect topological signatures of market stress regimes.

Can geometry predict a market crash? This project applies persistent homology to multi-dimensional equity return data to see if topological signatures can detect or predict market stress regimes.

The goal here is to bridge the gap between abstract algebraic topology research and practical quantitative finance by building a rigorous, end-to-end pipeline that extracts Betti number time series from S&P 500 returns and tests their true predictive power over market drawdowns.

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

**Persistent homology.** We compute the Vietoris-Rips filtration on $X_t$ using [Ripser](https://github.com/scikit-tda/ripser.py) which is used for filtration in homological percolation on lattice complexes. The output is a persistence diagram $\text{PD}(X_t)$ encoding the birth and death of topological features across scales.

---

## Results

### Key Findings

All five topological features were found to be stationary (ADF p < 0.05)
in their raw form. No differencing was required before Granger testing.

Essentially, all tests point towards there being no signal.

---

## Discussion

The efficient market hypothesis predicts that any predictable signal in
financial data will be arbitraged away once discovered. The results of
Gidea and Katz (2018), while visually compelling, were based on a small
number of events and did not apply the multiple comparison corrections
that are standard in statistical testing. A more rigorous test on a larger
dataset with formal correction finds no reliable signal.

---

## Conclusion

We find no statistically significant predictive signal from persistent
homology features of S&P 500 return point clouds over the period 2004–2024,
using multiple comparison correction across four classes of
statistical tests.

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

| `yfinance` | 0.2.51 |
| `ripser` | 0.6.8 |
| `persim` | 0.3.8 |
| `scikit-learn` | ≥1.1 |
| `pandas` | ≥1.5 |
| `numpy` | ≥1.23 |
| `scipy` | ≥1.9 |
| `statsmodels` | ≥0.13 |
| `matplotlib` | ≥3.6 |

---

## Related Work

This project is a inspired from my work on the repository [ATEAMS](https://github.com/apizzimenti/ATEAMS), which is a library for algebraic topology-enabled simulation of spin systems. ATEAMS applies persistent homology and finite-field linear algebra to study phase transitions in statistical physics. This project applies similar mathematical objects like filtrations, persistence diagrams, and Betti numbers to the geometry of financial markets.

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
