# Topology-Aware Real-Time Intrusion Detection System in Software-Defined Networks Using XGBoost

Source code, trained model, and environment configuration for the M.Tech thesis of the
same name. This repository is the authoritative, version-controlled home for the project;
the thesis document references it directly for reproducibility.

## Layout

| Path | Description |
|---|---|
| `controller/ids_controller.py` | Ryu SDN controller — OpenFlow flow-stats polling, feature extraction, XGBoost inference, automatic IPS enforcement |
| `topology/topology_runner.py` | Mininet Star/Mesh/Bus/Ring topologies, Scapy attack-injection harness, CSV ground-truth logger |
| `dashboard/live_dashboard_3.py` | Flask live comparative dashboard |
| `models/final_xgb_model_bundle.pkl` | Trained XGBoost classifier, scaler, and label encoder bundle |

## Running an experiment

Three processes, started in this order:

```bash
# Terminal 1 — controller
cd controller && PYTHONPATH=.. ryu-manager ids_controller.py

# Terminal 2 — topology + traffic injection
sudo PYTHONPATH=/home/mininet/mininet python3 topology/topology_runner.py --topo all --window 50

# Terminal 3 — dashboard
python3 dashboard/live_dashboard_3.py
```

Dashboard: http://127.0.0.1:5000

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the accompanying thesis (see repository "About" section
for the current citation once published).
