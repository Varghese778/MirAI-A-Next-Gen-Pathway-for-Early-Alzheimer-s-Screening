# MirAI Deployment Guide - Render.com

## Step-by-Step Instructions

### Step 1: Create GitHub Repository

1. Go to **github.com** → Sign in (or create account)
2. Click **"New repository"** (green button)
3. Name it: `mirai-alzheimers`
4. Set to **Public**
5. Click **"Create repository"**

---

### Step 2: Push Code to GitHub

Open terminal in `D:\mirai` folder and run:

```bash
git init
git add .
git commit -m "Initial commit - MirAI Alzheimer's Risk Assessment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mirai-alzheimers.git
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

---

### Step 3: Deploy on Render.com

1. Go to **render.com** → Sign up with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo `mirai-alzheimers`

#### Backend Service Settings:
| Setting | Value |
|---------|-------|
| Name | `mirai-backend` |
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT` |
| Instance Type | Free |

4. Click **"Create Web Service"** → Wait 5-10 mins for build

---

### Step 4: Deploy Frontend

1. Click **"New +"** → **"Static Site"**
2. Connect same repo
3. Settings:
   | Setting | Value |
   |---------|-------|
   | Name | `mirai-frontend` |
   | Root Directory | `frontend` |
   | Build Command | (leave empty) |
   | Publish Directory | `.` |

4. Click **"Create Static Site"**

---

### Step 5: Update Backend URL

After backend deploys, copy its URL (like `https://mirai-backend.onrender.com`).

Edit `frontend/js/api.js` line 9:
```javascript
: 'https://YOUR-ACTUAL-BACKEND-URL.onrender.com/api';
```

Push change to GitHub - Render will auto-redeploy.

---

## Your Live URLs

After deployment:
- **Frontend**: `https://mirai-frontend.onrender.com`
- **Backend**: `https://mirai-backend.onrender.com`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check requirements.txt syntax |
| 500 errors | Check Render logs for Python errors |
| CORS errors | Backend needs CORS middleware (already added) |
| Slow first load | Free tier sleeps after 15 mins inactivity |
