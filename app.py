import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_PATH = "PopulationDataset.csv"

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df["Year"].astype(str).str.match(r"^\d{4}$")].copy()
    df["Year"] = df["Year"].astype(int)
    df["Population_num"] = to_num(df["Population"])
    return df

def logistic_rhs(t, P, r, K):
    return r * P * (1 - P / K)

def rk4_step(f, t, y, h, r, K):
    k1 = f(t, y, r, K)
    k2 = f(t + h/2, y + h*k1/2, r, K)
    k3 = f(t + h/2, y + h*k2/2, r, K)
    k4 = f(t + h,   y + h*k3,   r, K)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_population(t_obs, P0, r, K, dt=0.05):
    t_min, t_max = float(np.min(t_obs)), float(np.max(t_obs))
    t_grid = np.arange(t_min, t_max + dt, dt)
    P_grid = np.zeros_like(t_grid, dtype=float)
    P_grid[0] = P0
    for i in range(1, len(t_grid)):
        P_grid[i] = rk4_step(logistic_rhs, t_grid[i-1], P_grid[i-1], dt, r, K)
    return np.interp(t_obs, t_grid, P_grid)

@st.cache_data
def fit_params(t_rel, P, P0):
    def loss(theta):
        r, K = theta
        if r <= 0 or K <= np.max(P)*0.9:
            return 1e30
        pred = simulate_population(t_rel, P0, r, K, dt=0.05)
        return np.mean((pred - P)**2)

    r0 = 0.02
    K0 = np.max(P) * 1.5
    res = minimize(loss, x0=np.array([r0, K0]),
                   bounds=[(1e-6, 1.0), (np.max(P), np.max(P)*100)])
    return float(res.x[0]), float(res.x[1])

st.title("Simulasi Pertumbuhan Populasi (Logistic ODE) — RK4")

df = load_data()
countries = sorted(df["Country"].dropna().unique().tolist())

country = st.selectbox("Pilih Country (untuk laporan: kunci 1 negara)", countries, index=countries.index("Indonesia") if "Indonesia" in countries else 0)

sub = df[df["Country"] == country][["Year", "Population_num"]].dropna().sort_values("Year")
sub = sub.rename(columns={"Population_num":"Population"})

t = sub["Year"].values.astype(float)
P = sub["Population"].values.astype(float)
t_rel = t - t[0]
P0 = float(P[0])

r_fit, K_fit = fit_params(t_rel, P, P0)

st.caption(f"Default parameter dari fitting: r={r_fit:.5f}, K={K_fit:.0f}")

r = st.slider("r (growth rate)", 0.0001, 0.2, float(r_fit), 0.0001)
K = st.slider("K (carrying capacity)", float(np.max(P)), float(np.max(P)*10), float(K_fit), float(np.max(P)*0.01))

P_sim = simulate_population(t_rel, P0, r, K, dt=0.05)

fig = plt.figure()
plt.plot(t, P, marker="o", label="Data")
plt.plot(t, P_sim, label="Simulasi (RK4)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title(f"{country} — Data vs Simulasi")
plt.legend()
st.pyplot(fig)
