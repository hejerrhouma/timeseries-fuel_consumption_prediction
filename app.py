import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO

# --- Chargement des données ---
@st.cache_data
def load_data():
    df = pd.read_csv("annual_df.csv")
    df = df.groupby("YEAR")["COMB (L/100 km)"].mean().reset_index()
    df.columns = ["Year", "Avg_Fuel_Consumption"]
    df.set_index("Year", inplace=True)
    return df

df = load_data()
ts = df["Avg_Fuel_Consumption"]

# --- Sidebar ---
st.sidebar.title("Configuration du modèle ARIMA")
p = st.sidebar.slider("Ordre p", 0, 5, 2)
d = st.sidebar.slider("Ordre d", 0, 3, 1)
q = st.sidebar.slider("Ordre q", 0, 5, 1)
steps = st.sidebar.slider("Années à prédire", 1, 10, 5)

# --- Titre principal ---
st.title("📈 Analyse et Prévision de la Consommation de Carburant au Canada")

# --- Statistiques descriptives ---
st.subheader("Statistiques de la série")
st.write(ts.describe())

# --- Visualisation des données ---
st.subheader("Consommation moyenne par année")
st.line_chart(ts)

# --- Décomposition ---
st.subheader("Décomposition de la série")
try:
    decomposition = seasonal_decompose(ts, model='additive', period=1)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title='Observée')
    decomposition.trend.plot(ax=ax2, title='Tendance')
    decomposition.seasonal.plot(ax=ax3, title='Saisonnalité')
    decomposition.resid.plot(ax=ax4, title='Résidus')
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.warning("Décomposition impossible : " + str(e))

# --- Modèle ARIMA ---
st.subheader("Modèle ARIMA")
try:
    model = ARIMA(np.log(ts), order=(p, d, q))
    result = model.fit()
    st.success("Modèle entraîné avec succès !")

    # --- Prévisions ---
    fc = result.get_forecast(steps=steps)
    f_log = fc.predicted_mean
    ci = fc.conf_int()

    f_level = np.exp(f_log)
    ci_lower = np.exp(ci.iloc[:, 0])
    ci_upper = np.exp(ci.iloc[:, 1])

    last_year = ts.index[-1]
    forecast_years = list(range(last_year + 1, last_year + 1 + steps))

    forecast_df = pd.DataFrame({
        "Year": forecast_years,
        "Forecast": f_level.values,
        "Lower_CI": ci_lower.values,
        "Upper_CI": ci_upper.values
    })

    # --- Graphique ---
    fig, ax = plt.subplots()
    ax.plot(ts, label="Historique")
    ax.plot(forecast_years, f_level, label="Prévision", marker="o")
    ax.fill_between(forecast_years, ci_lower, ci_upper, alpha=0.3, label="Intervalle 95%")
    ax.axvline(last_year, color="gray", linestyle="--", label="Fin des données")
    ax.set_title(f"Prévision ARIMA({p},{d},{q}) sur {steps} ans")
    ax.set_xlabel("Année")
    ax.set_ylabel("Consommation (L/100km)")
    ax.legend()
    st.pyplot(fig)

    # --- Export CSV ---
    buffer = BytesIO()
    forecast_df.to_csv(buffer, index=False)
    st.download_button("📥 Télécharger la prévision (.csv)", buffer.getvalue(), "forecast.csv", "text/csv")

    st.dataframe(forecast_df.set_index("Year"))

except Exception as e:
    st.error("Erreur dans le modèle ARIMA : " + str(e))
