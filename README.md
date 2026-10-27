# EstateLens Pune

College-project demo: estimates apartment rent and buying prices in Baner, Wakad, Hinjewadi, Kothrud, Hadapsar, and Viman Nagar.

## Run
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```
On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.
Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs

**Data notice:** `data/pune_properties.csv` contains synthetic demonstration data, not verified market listings. Predictions are for demonstrating the ML workflow only.
