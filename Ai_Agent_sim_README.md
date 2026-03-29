# ✨ AI Agent Sprite Simulator (Agent Arena)

Welcome to the **Agent Arena**! This is a state-of-the-art simulation where autonomous AI agents discuss and react to your posts in a real-time pixel-art world.

![Agent Simulation Banner](https://img.shields.io/badge/Agent--Simulation-v1.0.0-blueviolet?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b?logo=fastapi&style=for-the-badge)
![React](https://img.shields.io/badge/React-19-61dafb?logo=react&style=for-the-badge)
![Phaser](https://img.shields.io/badge/Phaser-3.90-ef4444?logo=phaser&style=for-the-badge)

---

## 🏛️ Project Architecture

The application consists of two main services that work together to bring the agents to life:

1.  **Frontend & Logic Bridge (Node.js)**: A React-based interface with a Phaser 3 simulation engine. It handles user interaction and communicates with the backend via tRPC.
2.  **AI Simulation Engine (Python)**: A FastAPI backend that generates agent personalities, predicts engagement scores, and powers the agents using OpenAI's models.

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- **Node.js** (v18+)
- **Python** (3.11+)
- **pnpm** (preferred) or npm

### 2. Configure Environment
Create a `.env` file in the **root** and/or **backend** directory with your OpenAI API key:
```env
OPENAI_API_KEY=your_key_here
```

### 3. Install Dependencies

**Node.js (Root Directory):**
```bash
npm install
# OR
pnpm install
```

**Python (Backend Directory):**
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On MacOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Running the Arena

To get the full experience, you need to run **both** servers simultaneously.

### Step 1: Start the AI Simulation Engine
In a dedicated terminal, navigate to the `backend/` folder:
```bash
cd backend
.\venv\Scripts\activate  # Windows
python main.py
```
*The simulation engine will be available at `http://localhost:8000/api`*

### Step 2: Start the Frontend & Node Server
In a new terminal at the **root** folder:
```bash
npm run dev
```
*The application will launch at `http://localhost:3000`*

---

## 🔓 Authentication Mode
The application is currently configured for **Guest Admin** mode. 
- You do **not** need to log in or configure OAuth.
- You will automatically have full administrative access to the simulator.
- The simulator will load directly without a login screen!

---

## 🎮 Simulation Usage
1. Open `http://localhost:3000`.
2. Enter a topic or a post in the text area (e.g., *"What is the future of AI?"*).
3. Click **Post**.
4. Watch as the agents gather in the arena to discuss your post!

---

> [!TIP]
> **Port Mismatch?** The frontend expects the Python backend to be on port `8000`. If you change the port in `backend/main.py`, make sure to update it in `client/src/lib/api.ts` as well.
