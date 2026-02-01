import operator
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
# Thêm thư viện vẽ đồ thị chẩn đoán
import seaborn as sns
import scipy.stats as stats

from deap import gp, base, creator, tools, algorithms
from functools import partial

# --- 1. SETTINGS ---
SEASONAL_PERIOD = 12
TEST_MONTHS = 24
POP_SIZE = 500 
NGEN = 50
CX_PB, MUT_PB = 0.7, 0.3
TREE_HEIGHT = 5
RANDOM_SEED = 42

# [CẤU HÌNH] LAG INPUT THEO Ý BẠN
MANUAL_LAGS = [1,3,12,13,25] 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 2. DATA PREP ---
CSV_PATH = "electricity_sales_clean.csv"

def get_data():
    try:
        df = pd.read_csv(CSV_PATH, dtype={'YYYYMM': str})
        df = df[~df['YYYYMM'].str.endswith('13')] 
        df['Date'] = pd.to_datetime(df['YYYYMM'], format='%Y%m')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        target_msn = df['MSN'].unique()[0] 
        full_data = df[df['MSN'] == target_msn][['Date', 'Value']].set_index('Date').sort_index()
        return full_data['Value'].dropna()
    except:
        print("-> Dummy Data generated.")
        dates = pd.date_range(start='2000-01-01', periods=200, freq='M')
        t = np.arange(200)
        vals = 100 + 0.5*t + 20*np.sin(2*np.pi*t/12) + np.random.normal(0, 5, 200)
        return pd.Series(vals, index=dates)

series_raw = get_data()
series_log = np.log(series_raw)
train_log = series_log.iloc[:-TEST_MONTHS]

# Sử dụng Lag do bạn chỉ định
found_lags = MANUAL_LAGS
print(f"-> Using MANUAL LAGS: {found_lags}")

# --- 3. PRIMITIVES (Dynamic Inputs) ---
def protectedDiv(x, y):
    return x / y if abs(y) > 1e-3 else 1.0

def activation_tanh(x):
    if x > 10: return 1.0
    if x < -10: return -1.0
    return np.tanh(x)

# Số lượng input = 4 (Error, L, T, S) + Số lượng Lag + 1 (Bias)
n_inputs = 4 + len(found_lags) + 1
pset = gp.PrimitiveSet("MAIN", n_inputs)

pset.renameArguments(ARG0='Error')
pset.renameArguments(ARG1='L1')
pset.renameArguments(ARG2='T1')
pset.renameArguments(ARG3='S_m')

for i, lag in enumerate(found_lags):
    arg_name = f'ARG{i+4}'
    pset.renameArguments(**{arg_name: f'Lag_{lag}'})

pset.renameArguments(**{f'ARG{n_inputs-1}': 'Bias'})

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(activation_tanh, 1)
pset.addEphemeralConstant("randW", lambda: round(random.uniform(-0.5, 0.5), 2))

# --- 4. EVALUATION LOGIC ---
def evaluate_incremental(individual, pset, train_vals, m, comp_type, detected_lags, l_func=None, t_func=None):
    func = gp.compile(expr=individual, pset=pset)
    n = len(train_vals)
    L, T, S = np.zeros(n), np.zeros(n), np.zeros(n)
    L[0] = train_vals[0]
    
    errors_sq = []
    
    for t in range(1, n):
        yt = train_vals[t]
        L1, T1 = L[t-1], T[t-1]
        S_m = S[t-m] if t >= m else 0.0
        
        y_hat = L1 + T1 + S_m
        curr_error = yt - y_hat
        
        dynamic_lags = []
        for lag_idx in detected_lags:
            val = train_vals[t - lag_idx] if t >= lag_idx else 0.0
            dynamic_lags.append(val)
            
        args = [curr_error, L1, T1, S_m] + dynamic_lags + [1.0]

        try:
            if comp_type == 'LEVEL':
                d = np.clip(func(*args), -1.0, 1.0)
                L[t] = L1 + T1 + d
                T[t], S[t] = 0, 0
            elif comp_type == 'TREND':
                d_L = np.clip(l_func(*args), -1.0, 1.0)
                L[t] = L1 + T1 + d_L
                d_T = np.clip(func(*args), -0.1, 0.1)
                T[t] = T1 + d_T
                S[t] = 0
            elif comp_type == 'SEASON':
                d_L = np.clip(l_func(*args), -1.0, 1.0)
                L[t] = L1 + T1 + d_L
                d_T = np.clip(t_func(*args), -0.1, 0.1)
                T[t] = T1 + d_T
                d_S = np.clip(func(*args), -0.3, 0.3)
                S[t] = S_m + d_S
                
            if t >= m: errors_sq.append(curr_error**2)
        except:
            return (1e9,)

    if not errors_sq: return (1e9,)
    return (np.sqrt(np.mean(errors_sq)),)

# --- 5. GP SETUP ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=TREE_HEIGHT))

# --- 6. RUNNER ---
def run_evolution(name, comp_type, l_f=None, t_f=None):
    print(f"\n--- Evolving {name} ---")
    toolbox.register("evaluate", partial(evaluate_incremental, pset=pset, 
                                         train_vals=train_log.values, 
                                         m=SEASONAL_PERIOD, comp_type=comp_type,
                                         detected_lags=found_lags,
                                         l_func=l_f, t_func=t_f))
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, CX_PB, MUT_PB, NGEN, halloffame=hof, verbose=False)
    print(f"Best RMSE: {hof[0].fitness.values[0]:.4f}")
    return gp.compile(hof[0], pset), hof[0]

if __name__ == "__main__":
    # Evolve 3 components
    l_func, _ = run_evolution("LEVEL", 'LEVEL')
    t_func, _ = run_evolution("TREND", 'TREND', l_f=l_func)
    s_func, _ = run_evolution("SEASON", 'SEASON', l_f=l_func, t_f=t_func)

    # --- 7. FINAL FORECAST & METRICS ---
    vals = series_log.values
    n = len(vals)
    L, T, S = np.zeros(n), np.zeros(n), np.zeros(n)
    L[0] = vals[0]
    preds = []
    
    for t in range(1, n):
        yt = vals[t]
        S_m = S[t-SEASONAL_PERIOD] if t >= SEASONAL_PERIOD else 0.0
        preds.append(L[t-1] + T[t-1] + S_m)
        
        curr_error = yt - (L[t-1] + T[t-1] + S_m)
        
        dynamic_lags = []
        for lag_idx in found_lags:
            val = vals[t - lag_idx] if t >= lag_idx else 0.0
            dynamic_lags.append(val)
        args = [curr_error, L[t-1], T[t-1], S_m] + dynamic_lags + [1.0]
        
        L[t] = L[t-1] + T[t-1] + np.clip(l_func(*args), -1.0, 1.0)
        T[t] = T[t-1] + np.clip(t_func(*args), -0.1, 0.1)
        S[t] = S_m + np.clip(s_func(*args), -0.3, 0.3)
        
    preds_real = np.exp(preds)
    actuals = np.exp(vals[1:])
    split_idx = len(train_log)
    rmse = np.sqrt(np.mean((actuals[split_idx-1:] - preds_real[split_idx-1:])**2))
    print(f"\nTEST RMSE: {rmse:,.2f}")

    # --- 8. VẼ BIỂU ĐỒ DỰ BÁO ---
    plt.figure(figsize=(14, 6))
    plt.plot(series_raw.index[1:], actuals, 'k-', alpha=0.3, label='Actual')
    plt.plot(series_raw.index[1:], preds_real, 'g--', label='GP-HW Model')
    plt.axvline(series_raw.index[split_idx], color='r', linestyle=':', label='Train/Test Split')
    plt.title(f"Forecast with Manual Lags {found_lags} | RMSE: {rmse:,.0f}")
    plt.legend()
    plt.show()

    # =========================================================================
    # --- 9. BỐN BIỂU ĐỒ CHẨN ĐOÁN (DIAGNOSTICS PLOTS) ---
    # =========================================================================
    
    # Lấy phần dư (residuals) trên tập TRAIN để kiểm tra xem mô hình học tốt chưa
    # (Phần dư = Thực tế - Dự báo)
    train_residuals = actuals[:split_idx-1] - preds_real[:split_idx-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Residual Diagnostics (Train Set) - Lags {found_lags}", fontsize=16)

    # 1. Residuals over Time
    # Kiểm tra xem sai số có ổn định quanh mức 0 không (không bị loe ra hay trôi đi)
    axes[0, 0].plot(train_residuals, color='blue', alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('1. Residuals over Time')
    axes[0, 0].set_ylabel('Error')

    # 2. Distribution (Histogram + KDE)
    # Kiểm tra xem sai số có phân phối chuẩn (hình chuông) không
    sns.histplot(train_residuals, kde=True, ax=axes[0, 1], color='green', stat='density')
    # Vẽ đường chuẩn lý tưởng để so sánh
    mu, std = stats.norm.fit(train_residuals)
    x = np.linspace(min(train_residuals), max(train_residuals), 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0, 1].plot(x, p, 'r--', linewidth=2, label='Normal Dist')
    axes[0, 1].set_title('2. Distribution (Histogram & KDE)')
    axes[0, 1].legend()

    # 3. Q-Q Plot (Normal Q-Q)
    # Kiểm tra các điểm có nằm trên đường chéo đỏ 45 độ không
    # Nếu nằm trên đường đỏ -> Sai số là chuẩn (Tốt)
    sm.qqplot(train_residuals, line='45', ax=axes[1, 0], fit=True)
    axes[1, 0].set_title('3. Normal Q-Q Plot')

    # 4. ACF Plot of Residuals
    # Kiểm tra xem còn quy luật nào bị bỏ sót không.
    # Nếu tất cả các cột nằm trong vùng xanh -> Tốt (White Noise)
    # Nếu còn cột nào vượt ra ngoài -> Mô hình vẫn chưa học hết quy luật đó.
    sm.graphics.tsa.plot_acf(train_residuals, lags=24, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title('4. ACF of Residuals (Autocorrelation)')

    plt.tight_layout()
    plt.show()
