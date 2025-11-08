# GenAI Surgery

Surgical AI app with Flask backend and React frontend.

## Backend
- Flask API with ML inference
- Models: Private on Hugging Face (`Akshaya1303/surgical-weights-private`)
- Deploy: Render.com

## Frontend
- React UI
- Deploy: Vercel

## Setup
1. Clone: `git clone https://github.com/Akshaya34673/GenAI_Surgery.git`
2. Backend: `cd backend && pip install -r requirements.txt && python app.py`
3. Frontend: `cd frontend && npm install && npm start`

## Environment Variables
- HF_TOKEN (Hugging Face)
- GOOGLE_CLIENT_ID/SECRET (OAuth)
- TWILIO_ACCOUNT_SID/AUTH_TOKEN (SMS)

## License
MIT