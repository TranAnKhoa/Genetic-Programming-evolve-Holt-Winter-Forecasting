[Report_Time_Series_Cat1_Group_7_Thur_78.pdf](https://github.com/user-attachments/files/24988618/Report_Time_Series_Cat1_Group_7_Thur_78.pdf)# 🧬 Evolutionary Time Series Forecasting with Genetic Programming

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Method](https://img.shields.io/badge/Method-Genetic_Programming-purple)
![Focus](https://img.shields.io/badge/Focus-Symbolic_Regression_&_AI-red)

## 📝 Overview

This project investigates an **AI-driven enhancement of the classical Holt–Winters time series model** using **Genetic Programming (GP)** and **Symbolic Regression**.

Instead of relying on fixed update equations and static smoothing behavior, the proposed approach allows the **forecasting equations themselves to evolve**, enabling a more adaptive and data-driven learning mechanism.

The performance is benchmarked against:
- Classical Holt–Winters
- SARIMA

---

## 📐 Problem Statement & Methodology

### 🔹 Method 1: Classical Holt–Winters
* **Category:** Statistical Time Series Model
* **Mechanism:** Predefined update equations for level, trend, and seasonality
* **Limitation:** Fixed structure, limited adaptability

### 🔹 Method 2: GP-based Holt–Winters (Proposed)
* **Category:** AI / Evolutionary Computation
* **Technique:** Genetic Programming + Symbolic Regression
* **Key Idea:**  
  Evolve alternative update equations for Holt–Winters instead of manually defining them.

GP searches over symbolic expression trees and optimizes forecasting accuracy via evolutionary operators (selection, crossover, mutation), effectively enabling **structure learning rather than parameter tuning**.

### 🔹 Method 3: SARIMA
* **Category:** Statistical Baseline
* **Role:** Seasonal autoregressive benchmark
* **Limitation:** Requires manual order selection and assumes linear dynamics

---

## 🧠 Why Genetic Programming?

From an AI and computer science perspective, Genetic Programming offers:

* **Structural Flexibility:** Learns functional forms, not just coefficients
* **Adaptive Dynamics:** Handles non-stationary and evolving patterns
* **Interpretability:** Produces explicit mathematical expressions (not black-box)
* **Hybrid Modeling:** Bridges classical time series analysis and symbolic AI

---

## 📺 Simulation Demo

Visual comparison of forecasting results across:
- Holt–Winters
- GP-based Holt–Winters
- SARIMA

🎥 **Demo Video:**  
👉 *https://github.com/user-attachments/assets/9261f885-1569-4e24-8bf9-11c5cc57a9ea*


---

## 🧮 Experimental Evaluation

All models are evaluated under identical data splits and error metrics.

Results show that:
- GP-based Holt–Winters captures nonlinear and adaptive dynamics
- Symbolic regression discovers meaningful update structures
- Performance is competitive and, in several cases, superior to classical baselines

📄 **Full Report:**  
👉 *[Uploading Report_Time_Series_Cat1_Group_7_Thur_78.pdf…]()*


---

## 🛠 Tech Stack

* **Language:** Python  
* **Core Methods:** Genetic Programming, Symbolic Regression  
* **Time Series Models:** Holt–Winters, SARIMA  
* **Libraries:** NumPy, Pandas  

---

## 👨‍💻 Author

**Group Project** 
**My task**
* **Focus:** Deploy an AI-driven Time Series Forecasting  
* **Domain:** Evolutionary Computation & Machine Learning  
