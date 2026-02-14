# Deployment Guide

## Part 1: Pipeline — Linux VM (Docker)

### Prerequisites
- Linux VM (Ubuntu 22.04+ recommended, 1 vCPU / 1 GB RAM minimum)
- Docker + Docker Compose installed
- Git installed

### Step 1: Install Docker (if not already)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo apt install docker-compose-plugin -y

# Verify
docker --version
docker compose version

# Log out and back in for group changes to take effect
exit
```

### Step 2: Clone the repo

```bash
git clone https://github.com/guernicastars/polymarket.git
cd polymarket/pipeline
```

### Step 3: Configure secrets

```bash
cp .env.example .env
nano .env
```

Fill in your ClickHouse Cloud credentials:

```
CLICKHOUSE_HOST=qgu31sw9py.germanywestcentral.azure.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=PKR4cvT.xwH9k
CLICKHOUSE_DATABASE=polymarket
```

### Step 4: Launch

```bash
# Build and start (detached)
docker compose up -d --build

# Verify it's running
docker compose ps

# Watch logs (live)
docker compose logs -f

# Check health
curl http://localhost:8080/health
```

### Step 5: Verify data is flowing

Wait 1-2 minutes after startup, then check ClickHouse:

```bash
# Quick check from inside the container
docker compose exec polymarket-pipeline python -c "
import clickhouse_connect, os
c = clickhouse_connect.get_client(
    host=os.environ['CLICKHOUSE_HOST'],
    port=int(os.environ['CLICKHOUSE_PORT']),
    username=os.environ['CLICKHOUSE_USER'],
    password=os.environ['CLICKHOUSE_PASSWORD'],
    secure=True
)
print('Markets:', c.command('SELECT count() FROM polymarket.markets'))
print('Prices:', c.command('SELECT count() FROM polymarket.market_prices'))
print('Trades:', c.command('SELECT count() FROM polymarket.market_trades'))
"
```

### Common Operations

```bash
# Stop pipeline
docker compose down

# Restart
docker compose restart

# Rebuild after code changes
git pull
docker compose up -d --build

# View recent logs
docker compose logs --tail 100

# Resource usage
docker stats polymarket-pipeline
```

### Troubleshooting

| Problem | Fix |
|---|---|
| Container keeps restarting | `docker compose logs --tail 50` — check for connection errors |
| Can't connect to ClickHouse | Verify `.env` credentials, check if ClickHouse Cloud service is running |
| High memory usage | Reduce batch sizes in `config.py` or lower the number of tracked markets |
| No data after 5 min | Check logs for rate limit errors or API connectivity issues |

---

## Part 2: Dashboard — Vercel

### Prerequisites
- Node.js 18+ installed locally
- Vercel account (free tier works)
- Vercel CLI: `npm i -g vercel`

### Step 1: Login to Vercel

```bash
vercel login
```

### Step 2: Deploy

```bash
cd polymarket/dashboard

# First deployment (will prompt for project setup)
vercel

# Answer the prompts:
# - Set up and deploy? Yes
# - Which scope? (your account)
# - Link to existing project? No
# - Project name: polymarket-signals (or whatever you want)
# - Directory with code? ./
# - Override settings? No
```

### Step 3: Set environment variables

Go to https://vercel.com → your project → Settings → Environment Variables.

Add these 4 variables (for **Production**, **Preview**, and **Development**):

| Variable | Value |
|---|---|
| `CLICKHOUSE_URL` | `https://qgu31sw9py.germanywestcentral.azure.clickhouse.cloud:8443` |
| `CLICKHOUSE_USER` | `default` |
| `CLICKHOUSE_PASSWORD` | `PKR4cvT.xwH9k` |
| `CLICKHOUSE_DB` | `polymarket` |

Or set them via CLI:

```bash
vercel env add CLICKHOUSE_URL production
# paste: https://qgu31sw9py.germanywestcentral.azure.clickhouse.cloud:8443

vercel env add CLICKHOUSE_USER production
# paste: default

vercel env add CLICKHOUSE_PASSWORD production
# paste: PKR4cvT.xwH9k

vercel env add CLICKHOUSE_DB production
# paste: polymarket
```

### Step 4: Production deploy

```bash
vercel --prod
```

Your dashboard is now live at the URL Vercel gives you (e.g., `polymarket-signals.vercel.app`).

### Step 5: Custom domain (optional)

```bash
# Add your domain
vercel domains add signals.yourdomain.com

# Follow DNS instructions Vercel provides
```

### Updating the Dashboard

```bash
# After code changes, just push to GitHub:
git add -A && git commit -m "update dashboard" && git push

# Vercel auto-deploys from main branch

# Or manual deploy:
vercel --prod
```

### Vercel Settings to Check

1. **Framework**: Should auto-detect Next.js
2. **Node.js Version**: 20.x (Settings → General)
3. **Function Region**: `iad1` or closest to your ClickHouse Cloud region (already set in `vercel.json`)
4. **Function Duration**: Free tier = 10s, Pro = 60s (sufficient for our queries)

### Troubleshooting

| Problem | Fix |
|---|---|
| Build fails | Run `npm run build` locally first to catch errors |
| 500 errors on dashboard | Check Vercel Function Logs (Vercel dashboard → Deployments → Functions) |
| Empty data | Make sure the pipeline is running and has had time to populate ClickHouse |
| Slow queries | ClickHouse Cloud may be in sleep mode — first query wakes it up (takes ~10s) |
| CORS errors | Shouldn't happen (server components), but check `next.config.ts` if it does |

---

## Deployment Order

1. **ClickHouse Cloud** — Already done (schema migrated)
2. **Pipeline on VM** — Start this first so data starts flowing
3. **Dashboard on Vercel** — Deploy once pipeline has been running for ~5 minutes

The pipeline needs to run for a few minutes before the dashboard has meaningful data to show. The market sync job runs immediately on startup, so markets metadata appears first. Prices and trades follow within 30-60 seconds.
