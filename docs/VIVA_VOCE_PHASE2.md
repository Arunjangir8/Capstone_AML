# Phase 2 & Deep Learning — Viva Voce Preparation Guide

### Project 7: Remaining Useful Life Prediction for Industrial Engines
**Course:** Advanced Machine Learning (AML) & Deep Learning (DL)  
**Dataset:** NASA C-MAPSS Turbofan Engine Degradation (FD001)

---

## 🎯 Top 25 Viva Questions & In-Depth Technical Answers

### 1. General & Problem Formulation

#### Q1. What is the fundamental difference between Phase 1 (AML) and Phase 2 (DL)?
**Answer:**
* **Phase 1 (AML Tabular):** We relied on manual, domain-specific feature engineering. We computed fixed 10-cycle summary statistics (moving average, moving std, trend slope, and baseline deviation `_dev`) and trained tabular algorithms (Linear Regression, Decision Tree, Random Forest).
* **Phase 2 (Deep Learning):** We feed raw, normalized 30-cycle multivariate sensor sequences $(30 \times 14)$ directly into sequence neural networks (PyTorch LSTM and 1D-CNN) without any hand-crafted features. The model automatically learns non-linear temporal degradation representations end-to-end.

#### Q2. Why is Remaining Useful Life (RUL) formulated as a regression problem rather than classification?
**Answer:**
RUL is an exact continuous numerical quantity (the number of operating cycles left before failure). Regression provides precise remaining cycle estimates (e.g., "14.2 cycles left"), enabling dynamic maintenance scheduling, spare parts inventory planning, and flight routing, which cannot be captured by coarse discrete categories.

#### Q3. Why is the RUL target capped at 125 cycles?
**Answer:**
In aircraft engines, physical degradation is undetectable during early operating life. An engine at cycle 20 with 250 cycles left exhibits identical baseline sensor telemetry to an engine at cycle 70 with 200 cycles left. Capping the target at 125 cycles creates a piecewise linear function: constant during healthy operation, and linearly decreasing during degradation. This prevents gradient instability and allows the model to concentrate its capacity on actual wear propagation.

---

### 2. Deep Learning & Recurrent Neural Networks (LSTM)

#### Q4. Explain the internal gating mechanism of an LSTM. Write the mathematical equations.
**Answer:**
An LSTM cell solves the vanishing gradient problem using a Constant Error Carousel (CEC) controlled by three continuous multiplicative gates:
1. **Forget Gate ($f_t$):** Decides what information to discard from the previous cell state $C_{t-1}$:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$) & Candidate State ($\tilde{C}_t$):** Decides what new information to store in the cell state:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$):** Combines forgotten past with new candidate memory via linear accumulation (addition instead of repeated multiplication):
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate ($o_t$) & Hidden State ($h_t$):** Filters the cell state to produce the visible representation:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

#### Q5. How does LSTM overcome the Vanishing Gradient problem of standard RNNs?
**Answer:**
In a standard RNN, the hidden state update is $h_t = \tanh(W h_{t-1} + U x_t)$. During backpropagation through time (BPTT), the gradient $\frac{\partial h_T}{\partial h_1}$ involves repeated matrix multiplications of $W^T \cdot \text{diag}(1 - \tanh^2(\cdot))$. If the eigenvalues of $W$ are $< 1$, the gradient decays exponentially to 0.
In an LSTM, the cell state update is **additive**: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. The derivative $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. If the forget gate is near 1, the gradient flows backward across dozens of time steps without exponential attenuation.

#### Q6. Why did you use Huber Loss (`SmoothL1Loss`) instead of Mean Squared Error (MSE)?
**Answer:**
Aircraft engine sensor telemetry contains high-frequency random noise and occasional sensor spikes.
* **MSE ($L_2$ loss):** Penalizes errors quadratically ($e^2$). An outlier sensor reading creates huge gradients ($\frac{\partial L}{\partial \hat{y}} = 2e$), causing gradient explosions or destabilizing recurrent weights.
* **Huber Loss:** Behaves quadratically for small errors ($|e| < 1$) and linearly for large errors ($|e| \ge 1$):
  $$L_\delta(e) = \begin{cases} \frac{1}{2}e^2, & |e| \le 1 \\ |e| - \frac{1}{2}, & |e| > 1 \end{cases}$$
  This provides the smooth convergence of MSE around the minimum while retaining the outlier robustness of MAE. Our ablation study proved that Huber Loss improved Test RMSE from 14.72 to 13.23.

---

### 3. Convolutional Sequence Modeling (1D-CNN)

#### Q7. How does a 1D Temporal CNN differ from an LSTM for sequence modeling?
**Answer:**
* **Processing Paradigm:** 1D-CNNs process all time steps in parallel by convolving 1D filter kernels along the time axis, whereas LSTMs process sequences sequentially step-by-step.
* **Receptive Field:** A 1D-CNN's receptive field is determined by kernel size and layer depth ($K \times L$). An LSTM has a theoretically unbounded receptive field via its recurrent hidden state.
* **Inductive Bias:** 1D-CNNs are exceptional at detecting local multi-channel patterns (e.g., sudden slope changes or high-frequency vibrations), but LSTMs are superior at monotonic accumulation of wear over long cycles. In our benchmark, LSTM achieved 13.23 RMSE vs 18.93 for 1D-CNN.

---

### 4. The Hybrid Approach & Additional Marks

#### Q8. What is the technical rationale for the Hybrid Approach? Why not just use LSTM alone?
**Answer:**
Simply averaging models without technical justification does not qualify as a valid hybrid approach. Our rationale is grounded in **bias-variance decomposition and complementary representation learning**:
1. **Random Forest's strength:** Operates on the 71 engineered features, where `s11_dev` (drift from the engine's initial healthy cycles) provides an explicit reference to the engine's individual baseline.
2. **LSTM's strength:** Operates on raw 30-cycle sequences, capturing continuous temporal acceleration and dynamic wear inertia.
3. **Orthogonal Errors:** Because their inductive biases and input representations are completely different, their prediction errors are weakly correlated. When linearly combined:
   $$\text{Var}(\hat{y}_{hybrid}) = w^2 \text{Var}(e_{RF}) + (1-w)^2 \text{Var}(e_{LSTM}) + 2w(1-w)\text{Cov}(e_{RF}, e_{LSTM})$$
   Because $\text{Cov}(e_{RF}, e_{LSTM})$ is small, total variance decreases, dropping Test RMSE from 11.44 (RF) and 13.23 (LSTM) to **10.99 cycles**, and the NASA Score to **170.06**.

#### Q9. How were the fusion weights determined?
**Answer:**
The fusion weight $w = 0.69$ was optimized strictly on the validation set using grid search and verified with non-negative least squares (NNLS), ensuring that the test set remained strictly unseen during parameter selection.

---

### 5. Evaluation, Validation & Data Leakage

#### Q10. Why must the train/validation split be performed by engine ID rather than random row splitting?
**Answer:**
An engine's sensor readings at cycle $t$ and cycle $t+1$ are practically identical. If rows were split randomly, cycle 50 of Engine 1 might be in the training set while cycle 51 of Engine 1 is in the validation set. The model would simply memorize the engine's identity rather than learning general degradation patterns, resulting in severe data leakage and artificially inflated scores. Splitting by engine ID (80 train engines, 20 validation engines) tests true generalization to unseen hardware.

#### Q11. Explain the NASA Asymmetric Scoring function. Why does it penalize late predictions more heavily?
**Answer:**
The formula is:
$$S = \sum_{i=1}^N s_i, \quad s_i = \begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \text{ (Early)} \\ e^{d_i / 10} - 1, & d_i \ge 0 \text{ (Late)} \end{cases}$$
where $d_i = \hat{y}_i - y_i$.
* If a model predicts $\hat{y} = 20$ when true $y = 10$ ($d = +10$, late prediction): penalty is $e^{10/10} - 1 = e^1 - 1 \approx 1.72$.
* If error is $d = +20$: penalty is $e^{20/10} - 1 = e^2 - 1 \approx 6.39$.
* In contrast, an early prediction $d = -20$ gives $e^{20/13} - 1 \approx 3.65$.
**Aviation Rationale:** Predicting that an engine has 20 cycles left when it actually fails in 5 cycles causes in-flight engine failure. Predicting it has 5 cycles left when it has 20 cycles left only leads to an early maintenance inspection.

#### Q12. What was observed in the Early vs Late degradation stage analysis?
**Answer:**
Errors are significantly higher in the late degradation stage ($RUL \le 100$) than in the early healthy stage ($RUL > 100$). In the early stage, sensor readings remain close to nominal baseline and the target is capped at 125, making prediction straightforward. In the late stage, sensor drift accelerates exponentially and wear kinetics vary across engines, increasing prediction variance. The Hybrid model maintained the lowest late-stage RMSE (11.10 cycles).

#### Q13. Why was sliding window length $W=30$ selected in the ablation study?
**Answer:**
In C-MAPSS FD001, test engines are truncated prior to failure. The shortest test engine trajectory has exactly 31 cycles. A window length of $W=30$ allows every single test engine to be evaluated at its final cycle without requiring artificial zero-padding or edge truncation. Ablation showed $W=15$ suffered from an insufficient receptive field (RMSE 15.22), while $W=45$ required zero-padding and degraded performance (RMSE 14.10).

