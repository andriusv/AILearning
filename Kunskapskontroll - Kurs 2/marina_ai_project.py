# ============================================================
# Montenegro Marina AI Project
# ============================================================
#
# Projektbeskrivning:
# Detta projekt använder maskininlärningsbaserade regressionsmodeller
# för att förutsäga beläggningsgraden i marinor i Montenegro.
#
# Systemet kombinerar:
# - syntetisk datagenerering
# - regressionsbaserad prediktion
# - modellutvärdering med RMSE
# - geografisk visualisering med Folium
# - planering av seglingsrutter
#
# Använda teknologier:
# Python, Pandas, NumPy, Scikit-learn, Folium och Geopy
#
# ============================================================

import pandas as pd
import numpy as np
import folium

from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# STEG 1 - SKAPANDE AV DATAUPPSÄTTNING
# ============================================================
# Denna sektion definierar projektets marina-data.
#
# Varje marina innehåller:
# - geografiska koordinater
# - marina-kapacitet
# - serviceinformation
# - bekvämlighetsindex
#
# Dessa variabler används senare som indatafunktioner
# för maskininlärningsmodellen.

# Lista över större marinor i Montenegro.
# Latitude och longitude används för kartvisualisering
# och beräkning av seglingsrutter.

marinas = [
    {"name": "Porto Montenegro", "city": "Tivat", "lat": 42.4304, "lon": 18.6990,
     "capacity": 450, "amenities_index": 10, "fuel": 1, "restaurants": 1},

    {"name": "Portonovi Marina", "city": "Herceg Novi", "lat": 42.4540, "lon": 18.5375,
     "capacity": 220, "amenities_index": 9, "fuel": 1, "restaurants": 1},

    {"name": "Dukley Marina", "city": "Budva", "lat": 42.2780, "lon": 18.8388,
     "capacity": 300, "amenities_index": 8, "fuel": 1, "restaurants": 1},

    {"name": "Marina Bar", "city": "Bar", "lat": 42.0931, "lon": 19.0944,
     "capacity": 380, "amenities_index": 6, "fuel": 1, "restaurants": 1},

    {"name": "Marina Kotor", "city": "Kotor", "lat": 42.4247, "lon": 18.7712,
     "capacity": 120, "amenities_index": 7, "fuel": 0, "restaurants": 1},
]

# ============================================================
# STEG 2 - GENERERING AV SYNTETISK DATA
# ============================================================
# Eftersom verklig marina-data är dyrt
# genereras syntetisk data för simuleringsändamål.
#
# Datasetet simulerar:
# - säsongsbaserade turistmönster
# - marina-popularitet
# - slumpmässig daglig variation
#
# Ett datapunkt genereras för varje dag på året
# för varje marina.

rows = []

for m in marinas:
    for day in range(1, 366):

        # Högre beläggning under sommarsäsongen
        # (ungefär dag 150 till dag 260)
        season = 80 if 150 <= day <= 260 else 50

        # Beläggningen påverkas av:
        # - säsongseffekt
        # - marina-bekvämligheter
        # - slumpmässig variation för att simulera verkligheten
        occupancy = (
            season
            + m["amenities_index"] * 2
            + np.random.randint(-10, 10)
        )

        # Säkerställer att beläggningen hålls mellan 10% och 100%
        occupancy = max(10, min(100, occupancy))

        rows.append([
            m["capacity"],
            m["amenities_index"],
            m["fuel"],
            m["restaurants"],
            day,
            occupancy
        ])

# Skapar DataFrame för maskininlärning
data = pd.DataFrame(rows, columns=[
    "capacity", "amenities_index", "fuel",
    "restaurants", "day_of_year", "occupancy"
])

# ============================================================
# STEG 3 - UPPDELNING AV TRÄNINGS- OCH TESTDATA
# ============================================================
# Datasetet delas upp i:
# - träningsdata (80%)
# - testdata (20%)
#
# Träningsdatan används för att träna modellerna,
# medan testdatan används för att utvärdera modellernas
# prestanda på tidigare osedd data.

X = data.drop("occupancy", axis=1)
y = data["occupancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# STEG 4 - MASKININLÄRNINGSMODELLER
# ============================================================
# Tre regressionsmodeller tränas och jämförs:
#
# 1. Linjär regression
#    - enkel modell baserad på linjära samband
#
# 2. Beslutsträd-regression
#    - trädstruktur som kan modellera icke-linjära samband
#
# 3. Random Forest-regression
#    - ensemblemetod som kombinerar flera beslutsträd
#    - minskar överanpassning och förbättrar noggrannhet

linear_model = LinearRegression().fit(X_train, y_train)

decision_tree_model = DecisionTreeRegressor(
    random_state=42
).fit(X_train, y_train)

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
).fit(X_train, y_train)

# ============================================================
# STEG 5 - RMSE-FUNKTION
# ============================================================
# RMSE (Root Mean Square Error) används för att mäta
# modellernas prediktionsfel.
#
# Ett lägre RMSE-värde innebär bättre modellprestanda.
# RMSE är ett vanligt mått för regressionsproblem.

def rmse(model, X, y):
    return np.sqrt(mean_squared_error(y, model.predict(X)))

# ============================================================
# STEG 6 - ANALYS AV ÖVERANPASSNING
# ============================================================
# Modellernas prestanda utvärderas på både:
# - träningsdata
# - testdata
#
# Stora skillnader mellan tränings- och test-RMSE
# kan indikera överanpassning.

models = {
    "Linear Regression": linear_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

rmse_results = {}

print("\n=== ANALYS AV ÖVERANPASSNING ===")

for name, model in models.items():

    train_rmse = rmse(model, X_train, y_train)
    test_rmse = rmse(model, X_test, y_test)

    rmse_results[name] = test_rmse

    print(f"\n{name}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

# ============================================================
# STEG 7 - AUTOMATISKT VAL AV BÄSTA MODELL
# ============================================================
# Modellen med lägst test-RMSE väljs automatiskt
# som slutlig prediktionsmodell.

best_model_name = min(rmse_results, key=rmse_results.get)

if best_model_name == "Linear Regression":
    best_model = linear_model
elif best_model_name == "Decision Tree":
    best_model = decision_tree_model
else:
    best_model = random_forest_model

print("\n=== VAL AV BÄSTA MODELL ===")
print(f"Bästa modell: {best_model_name}")
print(f"Lägsta Test RMSE: {rmse_results[best_model_name]:.2f}")

# ============================================================
# STEG 8 - INTERAKTIV KARTVISUALISERING
# ============================================================
# Folium används för att skapa en interaktiv karta
# som visar:
# - marina-positioner
# - förutsagd beläggning
# - antal lediga platser
#
# Markörfärger representerar beläggningsnivå:
# - grön   = låg beläggning
# - orange = medelhög beläggning
# - röd    = hög beläggning

future_day = 10

m_map = folium.Map(location=[42.35, 18.75], zoom_start=9)

for m in marinas:

    sample = pd.DataFrame([{
        "capacity": m["capacity"],
        "amenities_index": m["amenities_index"],
        "fuel": m["fuel"],
        "restaurants": m["restaurants"],
        "day_of_year": future_day
    }])

    # Predikterar framtida beläggning
    prediction = best_model.predict(sample)[0]
    prediction = max(0, min(100, prediction))

    occupied = int((prediction / 100) * m["capacity"])
    free_spots = m["capacity"] - occupied

    color = "red" if prediction > 80 else "orange" if prediction > 50 else "green"

    popup = f"""
    <b>{m['name']}</b><br>
    City: {m['city']}<br>
    Capacity: {m['capacity']} boats<br>
    <hr>
    Occupancy: {prediction:.1f}%<br>
    Occupied: {occupied}<br>
    <b>Free Spots: {free_spots}</b>
    """

    folium.Marker(
        [m["lat"], m["lon"]],
        popup=popup,
        tooltip=m["name"],
        icon=folium.Icon(color=color)
    ).add_to(m_map)

# ============================================================
# STEG 9 - VISUALISERING AV SEGLINGSRUTT
# ============================================================
# En fördefinierad seglingsrutt visualiseras mellan
# olika marinor med hjälp av geografiska koordinater.
#
# Rutten representeras som direkta linjer mellan punkter
# och tar inte hänsyn till verkliga navigationshinder.

route = [
    (42.4540, 18.5375),
    (42.4304, 18.6990),
    (42.4247, 18.7712),
    (42.2780, 18.8388),
    (42.0931, 19.0944)
]

folium.PolyLine(route, color="blue", weight=4).add_to(m_map)

# ============================================================
# STEG 10 - BERÄKNING AV TOTAL DISTANS
# ============================================================
# Geodesisk distans används för att beräkna den kortaste
# geografiska distansen mellan koordinater på jordens yta.

total_distance = 0

for i in range(len(route)-1):
    total_distance += geodesic(route[i], route[i+1]).km

print(f"\nTotal seglingsdistans: {total_distance:.1f} km")

# ============================================================
# STEG 11 - SPARA KARTAN
# ============================================================
# Den interaktiva kartan sparas som en HTML-fil
# och kan öppnas i valfri webbläsare.

m_map.save("montenegro_marinas_map.html")

print("\nKarta sparad: montenegro_marinas_map.html")
print("\nProjektet slutfördes framgångsrikt")