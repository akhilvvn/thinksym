Default from project root:
python dev_start.py

Manual mode:
Start API:
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload

Start Frontend:
cd frontend
npm install
npm run dev

Open:
http://localhost:3000
