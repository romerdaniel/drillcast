# 🔮 DrillCast — LSTM Sequence Drilling State Forecaster

**Ahead-of-the-bit drilling state forecasting using 30-ft temporal sequences.**

Part of the **Drilling ML Progression Series** built on the **Equinor Volve Field (Well F-9A, North Sea)** dataset:
1. 🔥 **[Axial Flare](https://axial-flare-copilot.streamlit.app/)** — Regression (*"How much ROP?"*)
2. 🛡️ **[DrillSense](https://drillsense-copilot.streamlit.app/)** — Classification (*"What is happening at the bit right now?"*)
3. 🔮 **DrillCast** — Sequence Deep Learning / Bi-LSTM (*"What state is coming in the next 10–30 ft?"*)

---

## 🧠 Architecture & Physics-Informed Temporal Features

- **Model:** Bidirectional LSTM (`128 → 64`) + Dense Sequence Head
- **Lookback Window:** 30 depth intervals (~30 ft sliding sequence window)
- **15 Temporal Features:**
  - **6 Raw Surface Signals:** `WOB_klbs`, `RPM`, `Torque_ftlbs`, `ROP_fthr`, `Flow_gpm`, `GR_API`
  - **6 Physics-Engineered Features:** `MSE_kpsi` (Teale 1965), `Mech_Efficiency`, `Bit_Aggressiveness`, `ROP_WOB`, `Torque_RPM`, `WOB_RPM`
  - **3 Sequence Rate-of-Change Derivatives:** `dROP/dDepth`, `dMSE/dDepth`, `dTorque/dDepth`
- **4 Drilling States:**
  - `0` — ✅ **Optimal**
  - `1` — ⚠️ **Suboptimal**
  - `2` — 🔴 **Stick-Slip Risk**
  - `3` — 🟣 **Bit Wear Risk**

---

## 👤 Author

**Romer Villalobos**  
Senior Drilling & Completion Engineer | ML Engineer | Houston, TX  
[LinkedIn](https://www.linkedin.com/in/romer-villalobos-2b33a653)
