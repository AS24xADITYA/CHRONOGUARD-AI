# ChronoGuard: Technical Briefing & Architecture Deep-Dive

### 1. Executive Summary & Problem Formalization

**ChronoGuard** is an open-source, zero-cost, air-gapped **temporal sequence forecasting engine** for network intrusion detection. 

Traditional Network Intrusion Detection Systems (NIDS)—ranging from signature-based tools like Snort/Suricata to tree-based machine learning models (Random Forests, XGBoost) and deep feed-forward networks—treat network flows as **independent and identically distributed ($i.i.d.$)** samples. They evaluate traffic on a static, point-in-time basis:
$$\hat{y}_t = f(\mathbf{x}_t)$$
Where $\mathbf{x}_t$ is the feature vector of a single network flow at time $t$. 

#### The Flaw of the Point-in-Time Paradigm:
Advanced Persistent Threats (APTs) and modern multi-stage adversaries execute attacks across distinct temporal phases (Reconnaissance $\rightarrow$ Credential Access $\rightarrow$ Lateral Movement $\rightarrow$ Exfiltration/Impact). Under an $i.i.d.$ assumption:
1. An isolated SYN packet or slow-rate port sweep is indistinguishable from standard network discovery or misconfigured DNS.
2. An adversary brute-forcing SSH credentials looks identical to transient authentication failures until compromise has already occurred.
3. Once the alert fires at the point of impact (e.g., ransomware payload detonation or DDoS flood), **the dwell time has already elapsed and the perimeter is breached**.

#### ChronoGuard’s Paradigm Shift:
ChronoGuard reformulates network defense as a **stochastic sequence forecasting problem**:
$$\mathbb{P}\left(\mathbf{y}_{t+k} \in \mathcal{C} \;\middle|\; \mathbf{W}_t = [\mathbf{x}_{t-W+1}, \dots, \mathbf{x}_t]\right)$$
Where:
- $\mathbf{W}_t$ is a sliding temporal window of $W = 10$ aggregated network flows.
- $k \in \{1, 2, 3\}$ is the **forecasting horizon** (anticipating escalation $k$ steps ahead).
- $\mathcal{C}$ is the set of **MITRE ATT&CK enterprise tactics**:
  $$\mathcal{C} = \{\text{Benign}, \text{Reconnaissance}, \text{Credential Access}, \text{Initial Access}, \text{Lateral Movement}, \text{Impact}\}$$

Rather than asking *"Is this flow malicious right now?"*, ChronoGuard answers:
> *"Given the micro-dynamics of the last $W$ flows, what is the probability that the network transitions into a high-severity stage within the next $k$ time windows?"*

---

### 2. Temporal Ingestion & Zero-Leakage Pipeline

```
Raw Telemetry (CIC-IDS-2017)
  │
  ▼
[Data Sanitizer] ── Clean whitespace, drop NaN/Inf, deduplicate SPAN mirrors
  │
  ▼
[Chronological Chunk Sampler] ── 65% Train | 10% Val | 25% Test (Per File)
  │
  ▼
[Sliding Flow Aggregator] ── Fixed-size buckets (W=10 flows, step=1)
  │
  ▼
[Feature Extractor] ── 16 engineered macro-kinetic flow features
  │
  ▼
[RobustScaler (5σ Clip)] ── Zero-leakage normalizer (fitted strictly on Train)
  │
  ▼
[Sequence Batcher] ── Tensor [BatchSize, SequenceLength=10, FeatureDim=16]
```

#### A. Telemetry Source: Real CIC-IDS-2017
ChronoGuard is trained and evaluated on raw capture extracts from the **Canadian Institute for Cybersecurity (CIC-IDS-2017)** dataset (~8.4 GB, 8 separate capture files spanning 5 consecutive business days):
- **Monday**: Normal baseline activity (strictly benign background traffic).
- **Tuesday**: Brute Force attacks (FTP-Patator, SSH-Patator).
- **Wednesday**: DoS (Slowloris, SlowHTTPTest, Hulk, GoldenEye) and Heartbleed exploit.
- **Thursday**: Web Attacks (Brute Force, XSS, SQLi) and Infiltration attempts.
- **Friday**: Botnet ARES, PortScans, and Volumetric DDoS (LOIC).

#### B. Zero-Data-Leakage Chronological Partitioning
In temporal sequence forecasting, standard random $k$-fold cross-validation or random train-test splitting introduces **catastrophic future-to-past data leakage** (look-ahead bias): future session statistics bleed into the training set.

ChronoGuard enforces strict chronological partitioning across all daily captures:
- **Training Set (Earliest 65%)**: Establishes baseline benign profiles, credential attacks, and DoS volumetric transitions across all capture days without lookahead bias.
- **Validation Set (Subsequent 10%)**: Independent chronological slice used exclusively for validation monitoring and checkpoint selection.
- **Test Set (Final 25%)**: Strictly untouched holdout evaluating out-of-sample temporal generalization and multi-horizon escalation forecasting across all 6 stages.

#### C. Ingestion Sanitization
The raw flow extractor outputs 83 raw statistical columns with numerous real-world anomalies handled by `src/data/clean.py`:
1. **Header Normalization**: Strips irregular leading/trailing ASCII whitespace across heterogeneous packet sniffers.
2. **Singular & Infinite Value Suppression**: In raw captures, flows with `Flow Duration = 0` yield division-by-zero errors in `Flow Bytes/s` and `Flow Packets/s`, generating `+Infinity` and `NaN`. These are safely mapped to domain extrema and imputed.
3. **SPAN Port Mirror Deduplication**: Dual-homed network taps often record bidirectional flows twice; identical timestamps and TCP tuples are deduplicated.

---

### 3. Feature Engineering: The 16-Dimensional Kinetic Representation

Raw packet headers contain static indicators (IPs, port numbers) that cause models to overfit to specific subnets rather than learning behavior. ChronoGuard strips IP addresses and aggregates raw flows into an engineered **16-dimensional kinetic feature space** $\mathbb{R}^{16}$ across each window $w_i$:

| Index | Feature Name | Mathematical / Cyber Significance | Threat Detection Sensitivity |
|---|---|---|---|
| 0 | `flow_duration_mean` | Mean session lifetime: $\frac{1}{N}\sum \Delta t_{\text{flow}}$ | Low in SYN scans; High in exfiltration tunnels |
| 1 | `flow_duration_std` | Standard deviation of session lifetimes | Machine-automated vs. human-driven browsing |
| 2 | `total_fwd_packets_mean` | Mean outbound packet count | Command & Control beaconing pulse checks |
| 3 | `total_bwd_packets_mean` | Mean inbound packet count | Data exfiltration volume or asymmetric drops |
| 4 | `total_fwd_pkts_std` | Variance of client-to-server burstiness | Brute-force burst consistency |
| 5 | `total_bwd_pkts_std` | Variance of server-to-client responses | Response size variance during reconnaissance |
| 6 | `fwd_packet_length_mean`| Mean payload size emitted by source | HTTP POST payload attacks vs. tiny probes |
| 7 | `fwd_packet_length_std` | Dispersion of forward payload lengths | Shellcode injection vs. static polling |
| 8 | `bwd_packet_length_mean`| Mean response payload from target | Database dump exfiltration or error cascades |
| 9 | `flow_bytes_per_sec_mean`| Mean velocity of raw data transfer | Saturation thresholds (DoS / DDoS) |
| 10 | `flow_iat_mean` | Mean Inter-Arrival Time between flows | Automated scripting cadence vs. human jitter |
| 11 | `flow_iat_std` | Variance of Inter-Arrival Times | High periodicity indicates C2 beacon heartbeat |
| 12 | `syn_flag_count_sum` | $\sum \mathbb{I}_{\text{SYN}}$ TCP handshakes initiated | TCP Half-Open / SYN flood reconnaissance |
| 13 | `rst_flag_count_sum` | $\sum \mathbb{I}_{\text{RST}}$ TCP teardowns/resets | Port closed response rates during network sweeps |
| 14 | `psh_flag_count_sum` | $\sum \mathbb{I}_{\text{PSH}}$ Push flag immediate dispatches| Interactive reverse shell communications |
| 15 | `ack_flag_count_sum` | $\sum \mathbb{I}_{\text{ACK}}$ TCP acknowledgments | Asymmetry ratio between SYN/ACK indicating scans |

#### Normalization:
A `RobustScaler` is trained on the full training partition (all classes) to preserve attack-feature dynamic range (preventing attack-class features from artificially saturating at the $\pm 5\sigma$ clipping boundaries as occurred when fit solely on benign flows). Features are normalized using their interquartile range ($IQR = Q_3 - Q_1$), followed by outlier clamping at $5\sigma$:
$$z = \text{clip}\left(\frac{x - \text{median}(X_{\text{train}})}{Q_3(X_{\text{train}}) - Q_1(X_{\text{train}})}, -5.0, 5.0\right)$$
This prevents extreme volumetric DDoS outliers from distorting recurrent gradients while preserving high-contrast separation across distinct attack families.

---

### 4. Neural Network Architecture: LSTM with Self-Attention Pooling

The core forecasting engine in `src/models/lstm.py` is a **Bidirectional/Deep 2-Layer Recurrent Neural Network with Additive Self-Attention Pooling**:

```
Input Sequence: X ∈ ℝ^[B x L=10 x D=16]
          │
          ▼
   [LSTM Layer 1]  ── Hidden Size = 64, Dropout = 0.2
          │
          ▼
   [LSTM Layer 2]  ── Hidden Size = 64, Dropout = 0.2
          │
          ▼
   Hidden States: H = [h₁, h₂, ..., h₁₀] ∈ ℝ^[B x 10 x 64]
          │
          ├──────────────────────────────────────────┐
          ▼                                          ▼
 [Attention MLP: uₜ = tanh(Wₐ hₜ + bₐ)]         [Hidden States H]
          │                                          │
          ▼                                          │
 [Softmax Normalization: αₜ = exp(uₜᵀ vₐ) / Σ exp]    │
          │                                          │
          ▼                                          ▼
   Attention Weights: α ∈ ℝ^[B x 10]        [Context Fusion: c = Σ αₜ hₜ]
                                                     │
                                                     ▼
                                            Context Vector: c ∈ ℝ^[B x 64]
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          ▼                                                     ▼
              [Multi-Class Stage Head]                               [Infiltration Risk Head]
            Dense(64 → 6) + Softmax                                    Dense(64 → 1) + Sigmoid
                          │                                                     │
                          ▼                                                     ▼
          P(Stage_{t+k} = c) ∈ ℝ⁶                                   P(Infiltration Escalation) ∈ [0, 1]
```

#### A. Recurrent Encoding
Given an input sequence of $L = 10$ temporal window vectors $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_L]$ where $\mathbf{x}_i \in \mathbb{R}^{16}$:
$$\mathbf{h}_t^{(1)} = \text{LSTM}_1\left(\mathbf{x}_t, \mathbf{h}_{t-1}^{(1)}\right)$$
$$\mathbf{h}_t^{(2)} = \text{LSTM}_2\left(\mathbf{h}_t^{(1)}, \mathbf{h}_{t-1}^{(2)}\right)$$
Each hidden state $\mathbf{h}_t \in \mathbb{R}^{64}$ encodes the sequential context up to step $t$.

#### B. Additive Self-Attention Pooling Mechanism
Standard RNNs either:
- Take only the final state $\mathbf{h}_L$ (which suffers from vanishing gradients over early sequence triggers).
- Take a uniform average $\frac{1}{L}\sum_{t=1}^L \mathbf{h}_t$ (which dilutes a 2-second port probe across 20 seconds of benign flows).

ChronoGuard implements a **parametric additive self-attention scoring function**:
$$\mathbf{u}_t = \tanh\left(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a\right), \quad \mathbf{u}_t \in \mathbb{R}^{32}$$
$$\alpha_t = \frac{\exp\left(\mathbf{u}_t^\top \mathbf{v}_a\right)}{\sum_{\tau=1}^L \exp\left(\mathbf{u}_\tau^\top \mathbf{v}_a\right)}, \quad \sum_{t=1}^L \alpha_t = 1$$
$$\mathbf{c} = \sum_{t=1}^L \alpha_t \mathbf{h}_t, \quad \mathbf{c} \in \mathbb{R}^{64}$$
- $\mathbf{W}_a \in \mathbb{R}^{32 \times 64}$ and $\mathbf{v}_a \in \mathbb{R}^{32}$ are learnable projection weights.
- $\alpha_t$ is the **attention scalar for time step $t$**. 
- $\mathbf{c}$ is the sequence summary context vector.

#### C. Dual Output Heads
1. **Multi-Class Stage Classifier**:
   $$\hat{\mathbf{y}}_{\text{stage}} = \text{Softmax}\left(\mathbf{W}_{\text{stage}} \mathbf{c} + \mathbf{b}_{\text{stage}}\right) \in \Delta^5$$
   Outputs the probability distribution over the 6 MITRE ATT&CK stages.
2. **Infiltration & Escalation Probability**:
   $$\hat{p}_{\text{infil}} = \sigma\left(\mathbf{w}_{\text{infil}}^\top \mathbf{c} + b_{\text{infil}}\right) \in [0, 1]$$
   Outputs a dedicated scalar representing the likelihood that the sequence transitions into an active compromised state.

---

### 5. Explainable AI (XAI) & Threat Attribution

A deep learning model in a Security Operations Center (SOC) is useless if treated as an uninterpretable black box. ChronoGuard provides **bilevel explainability**:

```
                  ┌────────────────────────────────────────────────────────┐
                  │               Bilevel Explainability Engine            │
                  └────────────────────────────────────────────────────────┘
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             ▼                                                                   ▼
[Macro-Level: Temporal Attribution]                               [Micro-Level: Feature Attribution]
• Source: Attention Vector α = [α₁, ..., α₁₀]                      • Source: Surrogate Kernel SHAP
• Answers: "WHEN did the attack sequence start?"                  • Answers: "WHICH metrics triggered the alert?"
• Visual: Dynamic timeline heatmap showing                         • Visual: Top-3 feature impact scores (e.g.,
  past window contribution to current forecast                     SYN surge, IAT drop, Payload variance)
```

1. **Temporal Attribution (Attention Vector $\boldsymbol{\alpha}$)**:
   The attention weights $\alpha_1, \dots, \alpha_{10}$ directly represent the temporal importance assigned by the neural network to each preceding time window. If a threat is forecasted at $t+1$, the SOC analyst can immediately inspect $\operatorname{argmax}_t(\alpha_t)$ to see the exact time window in the past that initiated the sequence.
2. **Feature Attribution (Surrogate Kernel SHAP / Gradient Sensitivity)**:
   To explain *which* specific network characteristics triggered the stage transition, ChronoGuard computes feature attributions over the 16 features:
   $$\phi_i = \mathbb{E}_{\mathbf{x}}\left[f(\mathbf{x}) \mid x_i\right] - \mathbb{E}[f(\mathbf{x})]$$
   Identifies whether the escalation was triggered by a surge in TCP flags (`syn_flag_count_sum`), flow inter-arrival collapse (`flow_iat_mean`), or payload asymmetry (`fwd_packet_length_std`).

---

### 6. Empirical Evaluation & Scientific Benchmarking

ChronoGuard was evaluated against an $L_2$-regularized **Multinomial Logistic Regression baseline** trained on the identical 16-dimensional feature space. Both models were trained and tested using a strict, leak-free **3-way chronological split (65% Train / 10% Validation / 25% Test)** across all daily captures of CIC-IDS-2017, guaranteeing zero future lookahead leakage while ensuring representation of all 6 MITRE ATT&CK stages. Both models were evaluated on $N = 1,805$ unseen sequence windows across **5 independent random seeds** (42, 123, 456, 789, 2024) using full-training `RobustScaler` normalization with $5\sigma$ clipping. All reported metrics represent empirical **Mean ± Standard Deviation** across the 5 evaluation seeds.

#### Test Set Class Distribution ($N = 1,805$ Windows):
- **Benign**: 1,202 windows (66.59%)
- **Reconnaissance**: 150 windows (8.31%)
- **Credential Access**: 112 windows (6.20%)
- **Initial Access**: 27 windows (1.50%)
- **Lateral Movement**: 24 windows (1.33%)
- **Impact**: 290 windows (16.07%)

*(Methodological Note on Class Distribution: The per-file chronological split samples the earliest 65% of each daily capture for training, the subsequent 10% for validation/early stopping, and reserves the final 25% exclusively for final evaluation, strictly preventing temporal data leakage while reflecting real operational flow distributions).*

#### Benchmark Comparison Table (5-Seed Empirical Mean ± Std):

| Evaluation Metric | Multinomial Logistic Regression (Baseline) | ChronoGuard (LSTM + Attention) | Architectural Analysis / Real-World Implication |
|---|---|---|---|
| **Attack Escalation Horizon $t+1$ Accuracy** | N/A (Static Single-Window) | **86.25% ± 6.17%** | Proactive temporal escalation forecasting 1 window ahead |
| **Attack Escalation Horizon $t+1$ F1 Score** | N/A (Static Single-Window) | **80.06% ± 7.14%** | Early warning capability before attack execution completes (Range: 69.75%–88.36%) |
| **Attack Escalation Horizon $t+2$ Accuracy** | N/A (Static Single-Window) | **86.15% ± 5.58%** | Sustained multi-step predictive forecasting across 2 windows |
| **Attack Escalation Horizon $t+2$ F1 Score** | N/A (Static Single-Window) | **79.82% ± 6.26%** | Stable detection lead time for automated mitigation triggers |
| **Attack Escalation Horizon $t+3$ Accuracy** | N/A (Static Single-Window) | **85.47% ± 5.31%** | Long-range temporal forecasting across 3 full windows |
| **Attack Escalation Horizon $t+3$ F1 Score** | N/A (Static Single-Window) | **78.80% ± 5.74%** | Deep temporal lookahead for security operations triage |
| **Overall Multi-Class Stage Accuracy** | 74.74% | **79.87% ± 8.07%** | ChronoGuard achieves higher mean accuracy with temporal sequence context |
| **Overall Multi-Class Macro F1** | 57.43% | **61.03% ± 10.47%** | Macro F1 range: 47.05% – 73.01% (varies by minority stage convergence) |
| **False Positive Rate (FPR)** | 25.21% | **19.98% ± 13.66%** | Temporal aggregation suppresses transient burst false alarms |
| **False Negative Rate (FNR)** | 21.39% | **15.85% ± 9.01%** | Lower miss rate on multi-window threat escalations |
| **Benign Stage F1 Score** | 80.63% | **84.55% ± 7.34%** | High benign discrimination across normal operational traffic |
| **Reconnaissance Stage F1 Score** | 83.38% | **83.52% ± 3.20%** | Both models achieve high fidelity (~83.5%) when features are scaled robustly |
| **Credential Access Stage F1 Score** | 0.00% | **35.01% ± 42.89%** | Baseline scores 0.00% (FTP vs SSH split); LSTM succeeds in 2/5 seeds (87–88% F1), fails in 3/5 (0.00% F1) |
| **Initial Access Stage F1 Score** | **82.61%** | 66.69% ± 30.71% | 4 of 5 seeds achieve ~82% F1; Seed 123 (Ep 12) collapses to Benign (6.9% F1) |
| **Lateral Movement Stage F1 Score** | 0.00% | 0.00% ± 0.00% | 0.00% F1 for both models: hard feature-separability ceiling without DPI/payload data |
| **Impact Stage F1 Score** | **97.96%** | 96.42% ± 1.11% | Near-perfect detection of high-velocity volumetric DoS/DDoS floods by both models |
| **Inference Latency (CPU, N=1,000 runs)** | **0.20 ms ± 0.07 ms** | **0.94 ms ± 0.29 ms** (P99: 1.68 ms) | Sub-millisecond wire-speed execution on standard commodity CPU (Batch=64: 0.04 ms/window) |
| **Model Footprint (Disk)** | **11.0 KB** | **304.3 KB** (75,945 params) | Fully self-contained edge deployment with zero cloud dependencies |

#### Critical Empirical Findings & Architectural Diagnoses

> [!IMPORTANT]
> **1. True Headline Finding: Binary Escalation Forecasting across Future Horizons ($t+1 \dots t+3$)**
>
> The definitive, empirically validated strength of ChronoGuard is **binary multi-horizon attack escalation forecasting**:
> - At horizon $t+1$, ChronoGuard achieves **80.06% ± 7.14% F1** (Accuracy: **86.25% ± 6.17%**, Range: 69.75% – 88.36%) across all 5 seeds.
> - At horizon $t+2$, performance remains rock-solid at **79.82% ± 6.26% F1** (Accuracy: **86.15% ± 5.58%**).
> - At horizon $t+3$, lookahead performance holds at **78.80% ± 5.74% F1** (Accuracy: **85.47% ± 5.31%**).
>
> Static single-window models (like Logistic Regression, Random Forests, or XGBoost) are **fundamentally incapable of future lookahead forecasting** because they lack temporal recurrence. ChronoGuard's dual-layer LSTM and self-attention mechanism successfully aggregate historical flow trajectories ($W=10$ past windows) to alert SOC analysts that an attack is actively building 1 to 3 time steps before execution completes.

> [!NOTE]
> **2. Credential Access: The FTP/SSH Protocol Split and Bimodal Generalization**
>
> In the chronological 65/10/25 split of Tuesday traffic:
> - **Training Set (first 65%)**: Contains 5,931 FTP-Patator flows and only 16 SSH-Patator flows.
> - **Test Set (final 25%)**: Contains 2,288 SSH-Patator flows (112 window sequences) and 0 FTP-Patator flows.
>
> Because FTP (port 21) transmits plain-text commands with distinct packet sizes while SSH (port 22) uses encrypted ciphertexts with uniform packet distributions, the **static linear baseline scores exactly 0.00% F1**, misclassifying 95.5% (107 of 112) of SSH sequences as **Benign**.
>
> In contrast, the ChronoGuard LSTM displays **bimodal cross-protocol generalization**:
> - In **2 of 5 seeds** (Seed 42: 87.11% F1; Seed 456: 87.96% F1), the model successfully generalizes across protocols, identifying 85–88% of SSH sequences by learning temporal connection-retry cadences (IAT periodicity and SYN/RST ratios) that are protocol-agnostic.
> - In **3 of 5 seeds** (Seeds 123, 789, 2024), the model fails completely (**0.00% F1**), with 94%–100% of SSH sequences collapsing into Benign—exactly like the baseline.
> - Diagnostic analysis shows that successful seeds stopped early (Epochs 3 and 5) before cross-entropy gradients were overwhelmed by the majority class. Therefore, cross-protocol brute-force transfer is **not a guaranteed capability**, but a promising temporal mechanism subject to optimization sensitivity in small minority regimes.

> [!NOTE]
> **3. Initial Access (66.69% ± 30.71% F1) & Macro F1 Variance Diagnosis**
>
> Across the $N = 27$ held-out Initial Access test sequences (Heartbleed and Web Attacks):
> - In **4 of 5 seeds** (Seeds 42, 456, 789, 2024), the LSTM demonstrates strong generalization, averaging **81.6% F1** (Seeds 42, 456, and 789 cluster tightly at 88.5%, 85.2%, and 84.6% F1 with 82–85% detection rates).
> - In **1 of 5 seeds** (Seed 123), performance collapses to **6.90% F1**, with 81.5% (22 of 27) of sequences misclassified as **Benign**.
> - This mirrors the Credential Access dynamic: Seed 123 stopped late at **Epoch 12**, where prolonged cross-entropy optimization against a large benign majority eroded minority stage decision boundaries.
> - **Crucial Takeaway on Variance**: Initial Access and Credential Access are the **sole drivers** of the 10.47% standard deviation in overall Macro F1 (61.03% ± 10.47%). In contrast, high-volume stages remain remarkably stable across all 5 seeds: Reconnaissance achieves **83.52% ± 3.20% F1** and Impact achieves **96.42% ± 1.11% F1**.

> [!WARNING]
> **4. Lateral Movement: Confirmed Feature-Separability Ceiling**
>
> On **Lateral Movement** (CIC-IDS-2017 Infiltration and Botnet C2 beaconing), **both the Baseline and the ChronoGuard LSTM score exactly 0.00% F1 across all 5 seeds**.
> This is a confirmed, insurmountable **feature-separability ceiling**: in a purely statistical 16-dimensional flow-metadata space without Deep Packet Inspection (DPI), DNS telemetry, or HTTP URI/payload inspection, low-frequency C2 beaconing is statistically indistinguishable from legitimate background web browsing. Acknowledging this limitation is critical for honest engineering: host-level or payload-level telemetry is strictly required to detect stealthy C2 channels.

> [!TIP]
> **5. Normalization Sensitivity: StandardScaler vs. RobustScaler**
>
> Normalization choice dramatically alters stage classification:
> - Under `StandardScaler`, PortScan (Reconnaissance) variance was compressed to $\sigma \approx 0.10$ due to benign bandwidth outliers, collapsing Reconnaissance F1 to 0.00%.
> - Under `RobustScaler` (Median + IQR with $5\sigma$ clipping fitted on the full training set), Reconnaissance F1 jumped to **83.38%** for the baseline and **83.52% ± 3.20%** for the LSTM.
> - Across scaler configurations, overall Macro F1 ranges honestly between **47.7% and 61.0%**, demonstrating that data preprocessing and scaling fidelity are as consequential as neural architecture choices in network intrusion pipelines.

The accuracy-vs-horizon stability curve is recorded in `results/forecast_horizon.png`, illustrating consistent predictive accuracy across horizons $t+1$ through $t+3$.
---

### 7. Systems & Software Engineering Architecture

ChronoGuard is implemented as an end-to-end, zero-cost, self-contained application designed to run on-premise without external dependencies or cloud subscriptions.

```
                                  Client Browser
                                        │
                         (HTTP / WebSocket-free Polling)
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ChronoGuard Flask Application                            │
│                                                                                  │
│  [REST Controllers]                                                              │
│  • GET  /            ── Landing Page (Hero, Model Architecture, Architecture)    │
│  • GET  /upload      ── Telemetry Ingestion (Drag-and-Drop / Demo Samples)       │
│  • POST /upload      ── Job Dispatcher (Asynchronous Background Thread)          │
│  • GET  /status/<id> ── Polling Endpoint (Progress, Window Count, Job State)     │
│  • GET  /results/<id>── SOC Command Center (Charts, Timelines, Mitre Badges)    │
│  • GET  /history     ── Audit Log (Past Analyses, Severity Indicators)           │
│  • GET  /api/health  ── Microservice Liveness & System Diagnostics               │
└───────────────────────┬──────────────────────────────────┬───────────────────────┘
                        │                                  │
                        ▼                                  ▼
         ┌──────────────────────────────┐   ┌──────────────────────────────┐
         │     SQLite Database          │   │  Inference Pipeline Engine   │
         │     (instance/chronoguard.db)│   │  (src/inference/pipeline.py) │
         ├──────────────────────────────┤   ├──────────────────────────────┤
         │ • AnalysisJob Table          │   │ • pandas / numpy cleaning    │
         │ • WindowPrediction Table     │   │ • RobustScaler feature scale │
         └──────────────────────────────┘   │ • PyTorch CPU model forward  │
                                            │ • Attention extraction       │
                                            └──────────────────────────────┘
```

#### A. Asynchronous Processing Worker
When a multi-megabyte CSV is uploaded via `/upload`:
1. The request thread registers an `AnalysisJob` in the SQLite database with `status = "queued"`.
2. A background worker (`concurrent.futures.ThreadPoolExecutor` or `threading.Thread`) is spawned, instantly returning an HTTP 302 redirect to the user.
3. The frontend initiates non-blocking progress polling against `/status/<job_id>`.
4. The background thread executes:
   - Chunked CSV parsing and NaN/inf sanitization.
   - Sliding window aggregation ($W=10$).
   - Matrix transformation and scaling.
   - PyTorch tensor forward pass on CPU.
   - Database batch persistence of each window's predictions and attention scores.
   - Job status update to `"done"`.

#### B. Database Schema (`src/db/models.py`)
- **`AnalysisJob`**:
  - `id`: UUID primary key.
  - `original_filename`: Telemetry source name.
  - `uploaded_at`: ISO timestamp.
  - `status`: Lifecycle state (`queued`, `running`, `done`, `failed`).
  - `total_windows`: Total temporal sequence steps evaluated.
  - `overall_max_infiltration_prob`: Highest detected escalation probability.
  - `dominant_stage`: Majority predicted MITRE ATT&CK classification.
- **`WindowPrediction`**:
  - Foreign key linked to `AnalysisJob.id`.
  - `window_index`: Chronological order index ($t_0, t_1, \dots$).
  - `benign_prob`, `recon_prob`, `cred_prob`, `init_prob`, `lateral_prob`, `impact_prob`: Raw softmax outputs.
  - `infiltration_prob`: Sigmoid risk output.
  - `predicted_stage`: Canonical classification string.
  - `attention_weight`: Attention scalar assigned to this window by the self-attention layer.

#### C. Air-Gapped Frontend Architecture
To satisfy defense and SCADA requirements where production SOC networks have **zero internet access**:
- **Offline Vendored Libraries**: 
  - Tailwind CSS (`web/static/vendor/tailwind.min.js`)
  - Chart.js (`web/static/vendor/chart.min.js`)
  - Alpine.js (`web/static/vendor/alpine.min.js`)
- **Visual Design**: High-contrast, military-grade dark SOC interface (`#070B14` canvas, `#0B1120` cards, `#22D3EE` cyan accentuation, `#EF4444` danger warnings).
- **Interactive Visualizations**:
  - Multi-axis Kill-Chain Escalation Timeline (temporal progression of stage probabilities).
  - Attention Heatmap (visualizing model focus across sequence history).
  - MITRE ATT&CK Radar Profile (distribution of observed tactics).

---

### 8. Architectural Rigor & Honest Limitations

ChronoGuard adheres to strict academic and engineering honesty:
1. **Sequence Forecasting vs. Reinforcement Learning**: 
   ChronoGuard models $\mathbb{P}(\mathbf{y}_{t+k} \mid \mathbf{W}_t)$. It is a **predictive sequence forecaster**, not a generative agent or autonomous response simulator. It does not hallucinate dynamic attacker responses; it detects the statistical signature of temporal build-up.
2. **Fixed-Size Sliding Windows**:
   Flows are bucketed by count ($W=10$) rather than strict wall-clock time. This ensures stability during bursts, though low-and-slow attacks spanning several hours require hierarchical time-decay windows (planned for v2.0).
3. **Encrypted Telemetry**:
   ChronoGuard operates strictly on **flow-level metadata** (packet sizes, inter-arrival times, TCP flags, duration). It does not perform Deep Packet Inspection (DPI) and is therefore completely agnostic to payload encryption (TLS 1.3, HTTPS, SSH tunnels).

---

The ChronoGuard repository contains the complete, self-contained system:
- **`src/data/`**: Ingestion, cleaning, canonical MITRE mapping, and sliding window aggregation algorithms.
- **`src/models/`**: PyTorch 2-layer LSTM with Additive Self-Attention Pooling, plus Baseline Logistic Regression.
- **`src/inference/pipeline.py`**: Zero-skew real-time inference orchestrator with attention extraction.
- **`src/db/models.py`**: SQLAlchemy schema for jobs and window predictions.
- **`web/`**: Dark SOC Jinja2 templates, custom CSS, and vendored offline UI engines.
- **`tests/`**: Complete 17-test validation suite verifying pipeline math, label bi-directionality, and API endpoints.
- **`models/`**: Serialized PyTorch model (`chronoguard_lstm.pt`), baseline (`baseline_lr.pkl`), and normalizer (`scaler.pkl`).
- **`data/samples/`**: Authentic sample slices extracted from CIC-IDS-2017 for instant single-click live demonstration.
