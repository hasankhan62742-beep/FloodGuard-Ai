# 🌊 FloodGuard AI — South Punjab Early Warning System

AI-powered flood early warning and disaster response prototype for South Punjab, Pakistan.

## Modules
- Village-level flood risk prediction (XGBoost)
- Satellite-based flood detection (U-Net segmentation)
- Interactive risk mapping (Folium)
- Real-time what-if scenario simulation
- Evacuation route planning (NetworkX / Dijkstra)
- Rescue resource dispatch optimization
- Time-to-flood forecasting (LSTM)
- Infrastructure vulnerability mapping
- Economic damage estimation

## Status
Prototype — built on historical flood-pattern and open-data structures.
Production deployment requires integration with live PMD/FFD/NDMA data feeds.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder Structure
```
FloodGuard-AI/
├── app.py
├── requirements.txt
├── models/
│   ├── risk_model.pkl
│   ├── unet_flood_model.h5
│   ├── lstm_flood_model.h5
│   └── damage_models.pkl
└── data/
    ├── villages_data.csv
    ├── map_data.csv
    ├── infrastructure_data.csv
    ├── safe_zones_data.csv
    └── rescue_resources.csv
```

## Developed By
Qadeer Automations
