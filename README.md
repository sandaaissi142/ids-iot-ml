\# IoT Intrusion Detection with Machine Learning



Network intrusion detection for IoT traffic using the \*\*ToN-IoT\*\* dataset, with a focus on \*\*rigorous evaluation\*\* (no data leakage) and \*\*lightweight models\*\* suited to embedded deployment.



\## Highlights

\- 10 traffic classes: normal + 9 attack types (backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss)

\- \*\*Data leakage avoided\*\*: 105,710 duplicate rows removed (50% of the raw file) before a stratified 80/20 split

\- IP addresses and ports excluded, so models learn traffic behaviour, not machine identities

\- Random Forest and XGBoost compared; lightweight variants measured for size and speed



\## Results (test set, 21,067 samples)



| Model | Accuracy | Macro F1 | F1 mitm |

|---|---|---|---|

| Random Forest | 0.9793 | 0.9590 | 0.7619 |

| XGBoost | 0.9810 | 0.9613 | 0.7686 |



!\[Class distribution](results/repartition\_types.png)

!\[XGBoost confusion matrix](results/confusion\_xgb.png)



\### Lightweight models (15 features)



| Model | Accuracy | Macro F1 | F1 mitm | Size | Time / 1000 predictions |

|---|---|---|---|---|---|

| RF full (52 features) | 0.9793 | 0.9590 | 0.7619 | \~106 MB | 25.3 ms |

| RF light | 0.9735 | 0.9462 | 0.6535 | \~4.8 MB | 3.8 ms |

| Decision tree | 0.9543 | 0.9192 | 0.5135 | 65 KB | 0.3 ms |



Compression costs accuracy mostly on the rarest class (`mitm`).



!\[Feature importance](results/importance\_rf.png)



\## Pipeline

1\. `src/explore.py`, `src/labels.py`, `src/inspect\_columns.py`: data exploration

2\. `src/preprocessing.py`: drops targets, IPs/ports, near-empty columns (>99% missing) and high-cardinality fields, then one-hot encoding (52 features)

3\. `src/split.py`: deduplication and stratified train/test split

4\. `src/train\_rf.py`, `src/train\_xgb.py`: model training and comparison

5\. `src/interpret.py`: feature importance and error analysis

6\. `src/light\_model.py`: lightweight models



\## Run it

```bash

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

```

Download `train\_test\_network.csv` (ToN-IoT, network data) into `data/`, then:

```bash

python src/preprocessing.py

python src/split.py

python src/train\_rf.py

python src/train\_xgb.py

python src/interpret.py

python src/light\_model.py

```



\## Limitations

\- ToN-IoT is lab-generated traffic: scores will be lower on real networks.

\- Random split, not time-based.

\- Model sizes and timings are measured in Python on a PC, not on a microcontroller.

\- `mitm` (about 1% of the data) remains the weakest class.



\## Dataset and citation

ToN-IoT, UNSW Canberra (Dr Nour Moustafa). Free for academic research; commercial use requires the author's permission.

Moustafa, N. "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON\_IoT datasets." \*Sustainable Cities and Society\* (2021).



\## Next steps

Export the decision tree to C for a microcontroller, and test on live traffic.

