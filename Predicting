import operator
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from deap import gp, base, creator, tools
from functools import partial
import os
import seaborn as sns
import scipy.stats as stats
import sys

# --- 1. HARDCODED SETTINGS & EQUATIONS ---
SEASONAL_PERIOD = 12
TEST_MONTHS = 125

# VUI LÒNG DÁN 3 CÔNG THỨC CUỐI CÙNG CỦA BẠN VÀO ĐÂY:
# SỬ DỤNG CÁC PHƯƠNG TRÌNH TỪ LẦN CHẠY TIẾN HÓA THÀNH CÔNG (GOOD RESULT)
EXPR_L_FINAL = "mul(sub(add(sub(ARG0, ARG8), sub(ARG0, mul(0.41, ARG8))), mul(ARG8, sub(add(ARG6, 0.27), sub(ARG0, 0.27)))), add(mul(add(add(ARG8, ARG0), sub(ARG8, ARG8)), sub(add(ARG8, 0.27), sub(ARG3, 0.27))), 0.27))"
EXPR_T_FINAL = "sub(protectedDiv(sub(sub(ARG8, mul(ARG4, ARG5)), mul(add(ARG8, ARG8), mul(ARG4, ARG5))), sub(mul(add(ARG5, ARG3), protectedDiv(ARG8, -0.06)), sub(add(ARG7, ARG0), sub(ARG1, ARG8)))), ARG2)"
EXPR_S_FINAL = "mul(add(add(activation_tanh(ARG7), add(0.31, ARG2)), sub(sub(0.32, mul(ARG5, 0.14)), add(mul(ARG8, 0.45), sub(ARG3, -0.11)))), mul(sub(sub(0.32, mul(ARG8, ARG3)), add(ARG0, mul(ARG5, ARG8))), add(add(ARG0, mul(ARG8, 0.45)), mul(add(0.31, ARG6), sub(0.18, ARG8)))))"

# File paths (giữ nguyên)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "electricity_sales_clean.csv")
PLOT_FORECAST = os.path.join(SCRIPT_DIR, "plot_forecast_final.png")
PLOT_DIAGNOSTICS = os.path.join(SCRIPT_DIR, "plot_diagnostics_final.png")
LOG_FILE = os.path.join(SCRIPT_DIR, "training_log_forecast_only.txt")

# --- 2. SUPPORT FUNCTIONS (Cần cho gp.compile) ---
def log_print(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def protectedDiv(x, y):
    return x / y if abs(y) > 1e-3 else 1.0

def activation_tanh(x):
    if x > 10: return 1.0
    if x < -10: return -1.0
    return np.tanh(x)

def get_pset(allow_div=True):
    """
    Cấu hình ARG indices KHỚP VỚI MÔI TRƯỜNG EVOLUTION (12 inputs)
    0: Error, 1: L1, 2: T1, 3: Sm, 4: Sin_M, 5: Cos_M, 6: Hint_T, 7: C, 
    8: Err_Lag1, 9: Err_Lag3, 10: Err_Lag4, 11: Err_Lag12
    """
    pset = gp.PrimitiveSet("MAIN", 12) # <--- SỬ DỤNG 12 INPUTS
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(activation_tanh, 1)
    if allow_div: pset.addPrimitive(protectedDiv, 2)
    return pset

# --- 3. DATA LOADING ---
try:
    df = pd.read_csv(CSV_PATH, dtype={'YYYYMM': str})
    df = df[~df['YYYYMM'].str.endswith('13')]
    df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    target_msn = df['MSN'].unique()[0]
    full_data = df[df['MSN'] == target_msn][['Date', 'Value']].set_index('Date').sort_index()
    series_raw = full_data['Value'].dropna()
except Exception as e:
    log_print(f">>> ERROR LOADING DATA: {e}. Cannot continue without data.")
    sys.exit(1)

series_log = np.log(series_raw)
train_log = series_log.iloc[:-TEST_MONTHS]

# --- 4. MAIN FORECASTING BLOCK (Sử dụng công thức cứng) ---
if __name__ == "__main__":
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    log_print(">>> STARTING FORECASTING MODE (Evolution Skipped) <<<")

    # 1. Compile Final Functions
    # get_pset() PHẢI CÓ 12 INPUTS để các ARG index (như ARG8, ARG9,...) được ánh xạ đúng
    pset = get_pset() 
    try:
        l_func = gp.compile(gp.PrimitiveTree.from_string(EXPR_L_FINAL, pset), pset)
        t_func = gp.compile(gp.PrimitiveTree.from_string(EXPR_T_FINAL, pset), pset)
        # S_func KHÔNG CÓ protectedDiv
        s_func = gp.compile(gp.PrimitiveTree.from_string(EXPR_S_FINAL, get_pset(allow_div=False)), get_pset(allow_div=False)) 
    except Exception as e:
        log_print(f"ERROR: Failed to compile equations. Check ARG index usage. Error: {e}")
        sys.exit(1)
    
    # Setup States
    vals = series_log.values
    n = len(vals)
    L, T, S = np.zeros(n), np.zeros(n), np.zeros(n)
    L[0] = vals[0]
    preds = []
    
    # SỬ DỤNG BUFFER 12 PHẦN TỬ ĐỂ LẤY CÁC LAG ERROR (khớp với code Evolution)
    error_history = [0.0] * 12 
    
    months = np.arange(n) % 12
    sin_m = np.sin(2 * np.pi * months / 12)
    cos_m = np.cos(2 * np.pi * months / 12)

    # 2. Run Fit/Prediction Loop
    log_print("Running in-sample and out-of-sample prediction...")
    for t in range(1, n):
        S_m = S[t-SEASONAL_PERIOD] if t >= SEASONAL_PERIOD else 0.0
        y_hat = L[t-1] + T[t-1] + S_m
        preds.append(y_hat)
        
        curr_error = vals[t] - y_hat
        Hint_T = L[t-1] - L[t-2] if t>=2 else 0.0
        
        # Lấy các Lag từ history (theo đúng index 12 inputs)
        err_lag1 = error_history[-1] 
        err_lag3 = error_history[-3] 
        err_lag4 = error_history[-4] 
        err_lag12 = error_history[-12]
        
        # NOTE: 12 ARGUMENTS PHẢI KHỚP VỚI get_pset (ARG0 -> ARG11)
        args = [curr_error, L[t-1], T[t-1], S_m, sin_m[t], cos_m[t], Hint_T, 1.0, 
                err_lag1, err_lag3, err_lag4, err_lag12]
        
        # Calculate Deltas
        d_L = np.clip(l_func(*args), -1.0, 1.0)
        d_T = np.clip(t_func(*args), -0.05, 0.05)
        d_S = np.clip(s_func(*args), -0.5, 0.5)
        
        L[t] = L[t-1] + T[t-1] + d_L
        T[t] = T[t-1] + d_T
        S[t] = S_m + d_S
        
        # Cập nhật lịch sử sai số (FIFO queue)
        error_history.pop(0)
        error_history.append(curr_error)
    
    # 3. Calculate Metrics and Forecast (Original Scale)
    preds_real = np.exp(preds)
    actuals_real = np.exp(vals[1:])
    split = len(train_log)
    
    actuals_train = actuals_real[:split-1]
    preds_train = preds_real[:split-1]
    actuals_test = actuals_real[split-1:]
    preds_test = preds_real[split-1:]
    train_rmse = np.sqrt(np.mean((actuals_train - preds_train)**2))
    test_rmse = np.sqrt(np.mean((actuals_test - preds_test)**2))
    train_mape = np.mean(np.abs(actuals_train - preds_train) / actuals_train) * 100
    test_mape = np.mean(np.abs(actuals_test - preds_test) / actuals_test) * 100

    log_print("\n--- PERFORMANCE METRICS (ORIGINAL SCALE) ---")
    log_print(f"TRAIN RMSE: {train_rmse:,.2f}")
    log_print(f"TEST RMSE : {test_rmse:,.2f}")
    log_print(f"TRAIN MAPE: {train_mape:,.2f} %")
    log_print(f"TEST MAPE : {test_mape:,.2f} %")
    
    # Dự báo 12 tháng (Logic ngoại suy)
    forecast_steps = 12
    L_last, T_last, S_last = L[-1], T[-1], S.copy()
    forecast_log = []
    
    # Dùng error_history cho dự báo (hoặc reset)
    error_forecast = [0.0] * 12 
    
    for h in range(1, forecast_steps + 1):
        month_idx = (len(vals) + h - 1) % 12
        sin_val = np.sin(2 * np.pi * month_idx / 12)
        cos_val = np.cos(2 * np.pi * month_idx / 12)
        S_m_forecast = S_last[len(vals) - SEASONAL_PERIOD + h - 1] if (len(vals) - SEASONAL_PERIOD + h - 1) < len(S_last) else 0.0
        y_hat_forecast = L_last + T_last + S_m_forecast
        forecast_log.append(y_hat_forecast)

        # Lấy các lag error cho dự báo (error = 0.0 trong dự báo)
        err_lag1_f = error_forecast[-1]
        err_lag3_f = error_forecast[-3]
        err_lag4_f = error_forecast[-4]
        err_lag12_f = error_forecast[-12]
        
        # 12 INPUTS CHO DỰ BÁO
        args_forecast = [0.0, L_last, T_last, S_m_forecast, sin_val, cos_val, T_last, 1.0, 
                         err_lag1_f, err_lag3_f, err_lag4_f, err_lag12_f] 
        
        d_L_f = np.clip(l_func(*args_forecast), -1.0, 1.0)
        d_T_f = np.clip(t_func(*args_forecast), -0.05, 0.05)
        # d_S không được tính trong dự báo ngoại suy, nó dùng S_m_forecast

        L_last = L_last + T_last + d_L_f
        T_last = T_last + d_T_f
        
        error_forecast.pop(0)
        error_forecast.append(0.0) # Sai số dự báo = 0.0

    forecast_real = np.exp(forecast_log)
    
    log_print(f"Forecast 12 months (Original Scale): {forecast_real}")
    
# ---------------- Plot Forecast with Confidence Intervals (Full Plot) ----------------
# ... (Phần vẽ Plot giữ nguyên, bạn có thể bổ sung các thư viện nếu cần)
import matplotlib.dates as mdates

dates = full_data.index
obs_dates = dates[1:]
hist_dates = obs_dates[:len(preds_real)]
last_date = dates[-1]

# ORIGINAL forecast dates
forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                               periods=len(forecast_real), freq='MS')

residuals_train = actuals_train - preds_train
sigma = np.std(residuals_train, ddof=1)
z = 1.96

forecast_upper = forecast_real + z * sigma
forecast_lower = np.maximum(0.0, forecast_real - z * sigma)

# ---------------------- FULL PLOT ----------------------
plt.figure(figsize=(14,6))

# TRAIN: light gray
plt.plot(dates[:len(train_log)], np.exp(vals[:len(train_log)]),
         label='Train (actual)', color='lightgray', linewidth=2)

# TEST actual
plt.plot(dates[len(train_log):], np.exp(vals[len(train_log):]),
         label='Test (actual)', color='gray', linewidth=2)

# In-sample prediction
plt.plot(hist_dates, preds_real, label='In-sample prediction',
         color='tab:blue', linewidth=1.7)

# Forecast
plt.plot(forecast_dates, forecast_real,
         label='Forecast (12 months)', color='tab:green', linewidth=2)

plt.fill_between(forecast_dates,
                 forecast_lower, forecast_upper,
                 color='tab:green', alpha=0.15, label='95% CI')

plt.title('Full Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(alpha=0.25)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))

plt.tight_layout()
plt.savefig(PLOT_FORECAST, dpi=200)
plt.show()


# ---------------------- ZOOM PLOT 2020–2026 ----------------------
PLOT_FORECAST_ZOOM = os.path.join(SCRIPT_DIR, "plot_forecast_zoom.png")

zoom_start = "2020-02-01"
zoom_end   = "2026-08-01"

zoom_ticks = pd.date_range("2020-02-01", "2026-08-01", freq="6MS")

plt.figure(figsize=(16,6))

# Actual
mask_zoom = (dates >= zoom_start) & (dates <= zoom_end)
plt.plot(dates[mask_zoom], np.exp(vals[mask_zoom]),
         label="Actual", color="lightgray", linewidth=2)

# Predictions
mask_pred_zoom = (hist_dates >= zoom_start) & (hist_dates <= zoom_end)
plt.plot(hist_dates[mask_pred_zoom], preds_real[mask_pred_zoom],
         label="Predicted", color="tab:blue", linewidth=1.7)

# Forecast
mask_fc_zoom = (forecast_dates >= zoom_start) & (forecast_dates <= zoom_end)
plt.plot(forecast_dates[mask_fc_zoom], forecast_real[mask_fc_zoom],
         label="Forecast", color="red", linewidth=2)

# CI
plt.fill_between(forecast_dates[mask_fc_zoom],
                 forecast_lower[mask_fc_zoom],
                 forecast_upper[mask_fc_zoom],
                 color="red", alpha=0.15, label="95% CI")

# Peak
peak_idx = np.argmax(forecast_real)
peak_date = forecast_dates[peak_idx]
peak_value = forecast_real[peak_idx]

plt.scatter(peak_date, peak_value, color="red", s=60, zorder=5)
plt.text(peak_date, peak_value,
         f"Peak: {peak_value:,.0f}",
         color="red", fontsize=11, ha="left", va="bottom")

plt.title("Forecast 2020–2026 (Zoomed)")
plt.xlabel("Month")
plt.ylabel("Value")
plt.grid(alpha=0.3)
plt.legend()

ax = plt.gca()
ax.set_xticks(zoom_ticks)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_FORECAST_ZOOM, dpi=220)
plt.show()

log_print(f"Peak Forecast Month: {peak_date.strftime('%Y-%m')}, Value = {peak_value:,.2f}")
# ---------------- PRINT FORECAST TABLE (12 months) ----------------
# Create DataFrame for pretty display
df_forecast = pd.DataFrame({
    "Date": forecast_dates.strftime("%Y-%m-%d"),
    "Forecast": forecast_real,
    "Lower": forecast_lower,
    "Upper": forecast_upper
})

# Format numbers with commas and 2 decimals
df_forecast["Forecast"] = df_forecast["Forecast"].map(lambda x: f"{x:,.2f}")
df_forecast["Lower"]    = df_forecast["Lower"].map(lambda x: f"{x:,.2f}")
df_forecast["Upper"]    = df_forecast["Upper"].map(lambda x: f"{x:,.2f}")

# Print header
print("\n--- KẾT QUẢ DỰ BÁO 12 THÁNG ĐẦU ---")
print(f"{'Date':<12} {'Dự báo (Duan)':>15} {'Lower':>15} {'Upper':>15}")
print("-" * 62)

# Print rows
for idx, row in df_forecast.iterrows():
    print(f"{row['Date']:<12} {row['Forecast']:>15} {row['Lower']:>15} {row['Upper']:>15}")
