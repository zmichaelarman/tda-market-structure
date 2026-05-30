# tda-market-structure

> Applying persistent homology to equity return data to detect topological signatures of market stress regimes.

Investigating whether topological features of S&P 500 return point clouds predict volatility regimes to create a bridge between algebraic topology research and quantitative finance. This project builds a rigorous end-to-end pipeline for extracting Betti number time series from equity returns and testing their predictive power over market drawdowns.

---

## Background

Most quantitative signals derived from equity data rely on statistical summaries like correlations, volatility estimates, momentum factors. This project asks a different question: does the **shape** of the return distribution, as measured by algebraic topology, carry information about market structure that conventional methods miss?

The mathematical framework comes from **topological data analysis (TDA)**, specifically persistent homology. Given a rolling window of daily returns across a universe of stocks, we construct a point cloud in high-dimensional return space and compute its Vietoris-Rips filtration. The resulting persistence diagrams encode how the topological structure of the market (aka its connected components, loops, and higher-dimensional cycles) evolves through time.

This project was inspired from my research conducted at the [Mason Experimental Geometry Lab](https://megl.science.gmu.edu/), where similar mathematical machinery was applied to spin systems and percolation models in [ATEAMS](https://github.com/apizzimenti/ATEAMS). The translation from statistical physics to financial markets is the core intellectual contribution of this work.

---

## Research Question

> Do Betti numbers and persistence statistics derived from S&P 500 return point clouds carry statistically significant predictive signal over volatility regimes, drawdowns, or market stress events?

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

**Persistent homology.** We compute the Vietoris-Rips filtration on $X_t$ using [Ripser](https://github.com/scikit-tda/ripser.py) which is the same class of filtration used in ATEAMS for homological percolation on lattice complexes. The output is a persistence diagram $\text{PD}(X_t)$ encoding the birth and death of topological features across scales.

---

## Results

### Stationarity

All five topological features were found to be stationary (ADF p < 0.05)
in their raw form. No differencing was required before Granger testing.

### Spearman Lead-Lag Correlations

No topological feature showed statistically significant lead-lag correlation
with any target variable at any tested horizon after FDR correction.

Raw correlations were small in magnitude (|ρ| < 0.08 in all cases),
consistent with noise rather than a genuine but weak signal.

### Mann-Whitney Stress Tests

No feature showed statistically significant distributional differences
between stress and non-stress periods after FDR correction.

Visual inspection reveals occasional spikes in H1 features 
around stress events, but these areneither systematic nor 
consistent across the five stress periods tested.

### Granger Causality

No feature was found to Granger-cause forward realized volatility or VIX
after FDR correction. P-values at the best lag ranged from 0.08 to 0.74,
none surviving correction.

---

## Discussion

### Why the Null Result is Not Surprising

The efficient market hypothesis predicts that any predictable signal in
financial data will be arbitraged away once discovered. The results of
Gidea and Katz (2018), while visually compelling, were based on a small
number of events and did not apply the multiple comparison corrections
that are standard in statistical testing. A more rigorous test on a larger
dataset with formal correction finds no reliable signal.

### Limitations and Alternative Hypotheses

The negative result is specific to the methodology employed here. Several
alternative approaches may yield different results and warrant investigation:

**Alternative point cloud constructions:**
- Correlation matrix entries as the point cloud (rather than return vectors)
- Sliding windows over intraday data rather than daily returns
- Alternative distance metrics (correlation distance rather than Euclidean)

**Alternative topological invariants:**
- Persistence images and persistence landscapes (vectorized representations)
- Mapper graph topology of the correlation network
- Cubical persistent homology of the correlation matrix as a 2D function

**Alternative universes:**
- Sector ETFs rather than individual stocks (reduces noise)
- International equity indices (tests generalization)
- Credit spreads and rates (different market microstructure)

**Alternative signal constructions:**
- Continuous position sizing based on z-score magnitude
- Combining topological features with standard volatility signals

### Connection to ATEAMS

The permutohedral lattice complex studied in ATEAMS provides a natural
alternative to the Vietoris-Rips complex used here. Permutohedral complexes
encode correlation structure differently (they do it through the geometry of the
permutohedron rather than pairwise distance thresholds), and may capture
topological features of the correlation matrix that Vietoris-Rips misses.

---

---

## Conclusion

We find no statistically significant predictive signal from persistent
homology features of S&P 500 return point clouds over the period 2004–2024,
using multiple comparison correction across four classes of
statistical tests.

This is a valid and informative scientific result. It does not rule out
the existence of topological signals in financial data — it rules out the
specific features and constructions tested here. The methodological
framework established by this project provides a reproducible baseline
for future investigation of alternative approaches.

All code, data, and results are publicly available in this repository.

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

This project is a inspired from my work on the repository [ATEAMS](https://github.com/apizzimenti/ATEAMS), which is a library for algebraic topology-enabled simulation of spin system. ATEAMS applies persistent homology and finite-field linear algebra to study phase transitions in statistical physics. This project applies similar mathematical objects like filtrations, persistence diagrams, and Betti numbers to the geometry of financial markets.

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

*Pipeline and results updated continuously as research progresses.*
