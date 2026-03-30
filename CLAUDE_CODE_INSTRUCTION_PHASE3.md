# CLAUDE CODE INSTRUCTION — PHASE 3: DEPLOY PHI-3 TRADING MODEL TO RASPBERRY PI 5

## Objective

Deploy the fine-tuned Phi-3 Mini 3.8B trading model (`phi3-trading-q4_k_m.gguf`, Q4_K_M, 2.2 GB) to a Raspberry Pi 5 (8 GB RAM, Hailo-8L NPU) and expose it as an OpenAI-compatible HTTP API. **All operations are executed remotely from the Windows development PC via SSH — no physical access to the Pi is required.**

---

## Prerequisites

- **Windows PC** with PowerShell and SSH client (built-in on Windows 10/11)
- **Raspberry Pi 5** running Raspberry Pi OS (Bookworm/Trixie), accessible on the local network
- Pi has completed Hailo AI HAT+ setup (steps 1–3 of the Hailo guide, before 3.1)
- The production GGUF exists at: `D:\Trading_AI_Agent\LoRa\phi3-trading-sft\phi3-trading-q4_k_m.gguf`

---

## IMPORTANT NOTES

- Every command targeting the Pi is executed via `ssh` or `scp` from the Windows terminal. **Do not assume local shell access to the Pi.**
- The SSH user and host are parameterized as `$PI_USER` and `$PI_HOST`. Resolve these from the operator before starting.
- If any SSH command fails with "connection refused" or "permission denied", stop and report — do not retry in a loop.
- The Hailo-8L NPU runs pre-compiled HEF models only. Our custom Phi-3 GGUF runs on the **ARM CPU via llama.cpp**. The NPU remains available for other inference layers (LSTM, etc.).

---

## Step 0 — Gather Connection Details

**ASK the operator for these values before proceeding:**

| Variable     | Description                          | Example              |
|-------------|--------------------------------------|----------------------|
| `PI_USER`   | SSH username on the Pi               | `pi` or `georg`      |
| `PI_HOST`   | Pi's IP address or hostname          | `192.168.1.50` or `trading-node.local` |
| `PI_PASS`   | Whether SSH key auth is set up, or if password will be entered manually | key / password |

Store them as shell variables for the session:

```powershell
$PI_USER = "pi"
$PI_HOST = "192.168.1.50"
$PI_TARGET = "$PI_USER@$PI_HOST"
```

### 0.1 — Verify SSH Connectivity

```powershell
ssh $PI_TARGET "echo 'SSH OK' && uname -a && free -h && df -h / | tail -1"
```

**Expected:** Kernel info showing `aarch64`, at least 4 GB free RAM, at least 5 GB free disk.

If SSH key auth is not configured and the operator wants passwordless access, set it up:

```powershell
# Generate key if none exists
if (!(Test-Path ~/.ssh/id_ed25519)) { ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519 -N '""' }

# Copy public key to Pi (will prompt for password once)
type $HOME\.ssh\id_ed25519.pub | ssh $PI_TARGET "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

Verify passwordless login works:

```powershell
ssh $PI_TARGET "echo 'Passwordless SSH OK'"
```

---

## Step 1 — Prepare the Pi

### 1.1 — Create Directory Structure

```powershell
ssh $PI_TARGET "mkdir -p ~/korgi/models ~/korgi/server ~/korgi/logs"
```

### 1.2 — Install Build Dependencies

```powershell
ssh $PI_TARGET "sudo apt update && sudo apt install -y build-essential cmake git libcurl4-openssl-dev"
```

### 1.3 — Check Available Resources

```powershell
ssh $PI_TARGET "free -h && nproc && cat /proc/cpuinfo | grep 'Model' | head -1"
```

**Expected:** 8 GB total RAM, 4 cores, Raspberry Pi 5.

---

## Step 2 — Transfer the GGUF Model

### 2.1 — Copy Model File to Pi

```powershell
scp "D:\Trading_AI_Agent\LoRa\phi3-trading-sft\phi3-trading-q4_k_m.gguf" "${PI_TARGET}:~/korgi/models/"
```

This transfers 2.2 GB. On a local gigabit network, expect ~30–60 seconds. On Wi-Fi, expect 2–5 minutes.

### 2.2 — Verify Transfer Integrity

Get the checksum on Windows first:

```powershell
$hash_win = (Get-FileHash "D:\Trading_AI_Agent\LoRa\phi3-trading-sft\phi3-trading-q4_k_m.gguf" -Algorithm SHA256).Hash
echo "Windows SHA256: $hash_win"
```

Then compare with the Pi:

```powershell
ssh $PI_TARGET "sha256sum ~/korgi/models/phi3-trading-q4_k_m.gguf"
```

**The hashes MUST match.** If they don't, re-transfer. Do not proceed with a corrupted model file.

---

## Step 3 — Build llama.cpp on the Pi

### 3.1 — Clone and Build

```powershell
ssh $PI_TARGET "git clone --depth 1 https://github.com/ggerganov/llama.cpp ~/korgi/llama.cpp"
```

```powershell
ssh $PI_TARGET "cd ~/korgi/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_AARCH64=ON && cmake --build build --config Release --target llama-server llama-cli -j 4"
```

**Note:** `-DGGML_CPU_AARCH64=ON` enables ARM NEON SIMD optimizations for the Pi 5's Cortex-A76 cores. The build takes ~5–10 minutes on the Pi.

### 3.2 — Verify Build

```powershell
ssh $PI_TARGET "~/korgi/llama.cpp/build/bin/llama-cli --version"
```

**Expected:** Version string with build info. If this fails, check the build output for errors.

---

## Step 4 — Test the Model (Interactive Smoke Test)

### 4.1 — Run a Single Inference

```powershell
ssh $PI_TARGET "cd ~/korgi/llama.cpp && ./build/bin/llama-cli -m ~/korgi/models/phi3-trading-q4_k_m.gguf -c 512 -n 128 --temp 0.7 -p '<|user|>\nWhat are the key indicators for identifying a bullish reversal pattern in BTC?<|end|>\n<|assistant|>\n' --no-display-prompt 2>/dev/null"
```

**Expected:** A coherent trading-domain response. Check that:
1. Output is in English (not garbled tokens)
2. Content references trading concepts from the training corpus
3. Generation completes without OOM or segfault

If the output is garbled or the model crashes, stop and report. Do not proceed to server setup.

### 4.2 — Measure Performance Baseline

```powershell
ssh $PI_TARGET "cd ~/korgi/llama.cpp && ./build/bin/llama-cli -m ~/korgi/models/phi3-trading-q4_k_m.gguf -c 512 -n 64 -p 'Explain market microstructure' 2>&1 | tail -5"
```

Note the **tokens/second** from the output. Expected range for Q4_K_M on Pi 5: **~5–12 tok/s** depending on context length. Record this for later comparison.

---

## Step 5 — Deploy the HTTP Server

### 5.1 — Create a Server Launch Script

```powershell
ssh $PI_TARGET @'
cat > ~/korgi/server/start_phi3.sh << 'SCRIPT'
#!/bin/bash
# Korgi Phi-3 Trading Model Server
# Serves OpenAI-compatible API on port 8080

MODEL_PATH="$HOME/korgi/models/phi3-trading-q4_k_m.gguf"
LOG_FILE="$HOME/korgi/logs/phi3-server.log"
PORT=8080
CTX_SIZE=2048
THREADS=$(nproc)

echo "[$(date)] Starting Phi-3 Trading Server on port $PORT" | tee -a "$LOG_FILE"
echo "[$(date)] Context: $CTX_SIZE | Threads: $THREADS" | tee -a "$LOG_FILE"

exec ~/korgi/llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    -c "$CTX_SIZE" \
    -t "$THREADS" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-prefix \
    2>&1 | tee -a "$LOG_FILE"
SCRIPT
chmod +x ~/korgi/server/start_phi3.sh
echo "Launch script created."
'@
```

### 5.2 — Create systemd Service (Auto-Start on Boot)

```powershell
ssh $PI_TARGET @'
sudo tee /etc/systemd/system/korgi-phi3.service > /dev/null << 'SERVICE'
[Unit]
Description=Korgi Phi-3 Trading Model Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/korgi/server
ExecStart=/home/$USER/korgi/server/start_phi3.sh
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits — prevent OOM from killing other services
MemoryMax=4G
MemoryHigh=3G

[Install]
WantedBy=multi-user.target
SERVICE
'@
```

**IMPORTANT:** The `$USER` in the service file needs to resolve to the actual Pi username. Fix it:

```powershell
ssh $PI_TARGET "sudo sed -i ""s|\`$USER|$USER|g"" /etc/systemd/system/korgi-phi3.service && cat /etc/systemd/system/korgi-phi3.service"
```

Verify the file looks correct (paths point to `/home/<actual_user>/korgi/...`).

### 5.3 — Enable and Start the Service

```powershell
ssh $PI_TARGET "sudo systemctl daemon-reload && sudo systemctl enable korgi-phi3.service && sudo systemctl start korgi-phi3.service"
```

### 5.4 — Verify Server is Running

Wait 5–10 seconds for model loading, then:

```powershell
ssh $PI_TARGET "sudo systemctl status korgi-phi3.service --no-pager -l | head -20"
```

**Expected:** `active (running)`.

Check the health endpoint:

```powershell
ssh $PI_TARGET "curl -s http://localhost:8080/health"
```

**Expected:** `{"status":"ok"}` or similar JSON response.

---

## Step 6 — End-to-End API Test

### 6.1 — Test from the Pi (localhost)

```powershell
ssh $PI_TARGET @'
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi3-trading",
    "messages": [
      {"role": "system", "content": "You are a trading analysis assistant specialized in stocks, commodities, and cryptocurrency markets."},
      {"role": "user", "content": "Analyze the significance of a death cross pattern in ETH/USD on the daily timeframe."}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }' | python3 -m json.tool
'@
```

**Expected:** A structured JSON response with a `choices[0].message.content` field containing a trading-relevant analysis.

### 6.2 — Test from Windows PC (Remote Access)

This confirms the server is accessible across the network:

```powershell
$body = @{
    model = "phi3-trading"
    messages = @(
        @{ role = "system"; content = "You are a trading analysis assistant." }
        @{ role = "user"; content = "What is the significance of increasing volume during a price breakout?" }
    )
    temperature = 0.7
    max_tokens = 256
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri "http://${PI_HOST}:8080/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
```

**Expected:** Same structured response, confirming the API is reachable from the Windows dev machine.

If this fails with a connection error, check the Pi firewall:

```powershell
ssh $PI_TARGET "sudo ufw status 2>/dev/null || echo 'ufw not active'"
ssh $PI_TARGET "sudo ufw allow 8080/tcp 2>/dev/null; echo 'Port 8080 opened'"
```

---

## Step 7 — Monitoring & Management (Remote)

### 7.1 — Useful Remote Commands

```powershell
# View live server logs
ssh $PI_TARGET "sudo journalctl -u korgi-phi3 -f"

# Check resource usage
ssh $PI_TARGET "ps aux | grep llama-server | grep -v grep"

# Check memory
ssh $PI_TARGET "free -h"

# Restart the server
ssh $PI_TARGET "sudo systemctl restart korgi-phi3"

# Stop the server
ssh $PI_TARGET "sudo systemctl stop korgi-phi3"

# View recent logs (last 50 lines)
ssh $PI_TARGET "sudo journalctl -u korgi-phi3 --no-pager -n 50"
```

### 7.2 — Create a Remote Health Check Script (Windows)

```powershell
ssh $PI_TARGET @'
cat > ~/korgi/server/health_check.sh << 'SCRIPT'
#!/bin/bash
# Quick health check for Korgi Phi-3 server

echo "=== Korgi Phi-3 Health Report ==="
echo "Timestamp: $(date)"
echo ""

# Service status
echo "--- Service Status ---"
systemctl is-active korgi-phi3 2>/dev/null && echo "Service: RUNNING" || echo "Service: DOWN"

# API health
echo ""
echo "--- API Health ---"
HEALTH=$(curl -s --max-time 5 http://localhost:8080/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "API: RESPONDING"
    echo "Response: $HEALTH"
else
    echo "API: NOT RESPONDING"
fi

# Resource usage
echo ""
echo "--- Resources ---"
free -h | grep Mem | awk '{print "RAM: " $3 " used / " $2 " total (" $7 " available)"}'
echo "CPU Load: $(cat /proc/loadavg | cut -d' ' -f1-3)"
echo "Disk: $(df -h / | tail -1 | awk '{print $3 " used / " $2 " total (" $5 " used)"}')"

# llama-server process
echo ""
echo "--- Process ---"
PID=$(pgrep -f llama-server)
if [ -n "$PID" ]; then
    echo "PID: $PID"
    echo "Uptime: $(ps -o etime= -p $PID | xargs)"
    echo "RSS: $(ps -o rss= -p $PID | awk '{printf "%.0f MB\n", $1/1024}')"
else
    echo "llama-server process NOT FOUND"
fi
SCRIPT
chmod +x ~/korgi/server/health_check.sh
echo "Health check script created."
'@
```

Run it anytime from Windows:

```powershell
ssh $PI_TARGET "~/korgi/server/health_check.sh"
```

---

## Step 8 — (Optional) Ollama Integration for Open WebUI

If you want the model to appear in Open WebUI alongside the Hailo-served models, you can also register it with a standard Ollama instance. This is **optional** — the llama.cpp server from Step 5 is the primary production endpoint.

### 8.1 — Install Standard Ollama (Separate from hailo-ollama)

```powershell
ssh $PI_TARGET "curl -fsSL https://ollama.ai/install.sh | sh"
```

**Note:** Standard Ollama listens on port 11434. The `hailo-ollama` server uses port 8000. They do not conflict.

### 8.2 — Create a Modelfile and Import

```powershell
ssh $PI_TARGET @'
cat > ~/korgi/models/Modelfile << 'MF'
FROM /home/$USER/korgi/models/phi3-trading-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>

SYSTEM """You are a trading analysis assistant specialized in stocks, commodities, and cryptocurrency markets. You provide analysis grounded in technical analysis, market microstructure, quantitative finance, and on-chain analytics."""

TEMPLATE """<|system|>
{{ .System }}<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""
MF
'@
```

Fix the `$USER` path:

```powershell
ssh $PI_TARGET "sed -i ""s|\`$USER|$USER|g"" ~/korgi/models/Modelfile"
```

Import the model:

```powershell
ssh $PI_TARGET "ollama create phi3-trading -f ~/korgi/models/Modelfile"
```

Test it:

```powershell
ssh $PI_TARGET "ollama run phi3-trading 'What is the Wyckoff accumulation pattern?' --nowordwrap"
```

### 8.3 — Connect Open WebUI

If Open WebUI is running (Step 4 of the Hailo guide), it should auto-detect the model on Ollama's port 11434. If the Open WebUI container was started with `OLLAMA_BASE_URL=http://127.0.0.1:8000` (hailo-ollama only), you'll need to add standard Ollama as a second connection in Open WebUI settings, or restart the container with both endpoints configured.

---

## Final State

After completing all steps, the Pi should have:

```
~/korgi/
├── models/
│   ├── phi3-trading-q4_k_m.gguf   # 2.2 GB — production model
│   └── Modelfile                    # (optional) Ollama model definition
├── server/
│   ├── start_phi3.sh               # Launch script
│   └── health_check.sh             # Health monitoring
├── llama.cpp/                       # Built inference engine
│   └── build/bin/
│       ├── llama-server             # HTTP server binary
│       └── llama-cli                # CLI testing binary
└── logs/
    └── phi3-server.log              # Server logs
```

**Services running:**
- `korgi-phi3.service` — llama.cpp server on port **8080** (auto-starts on boot)
- OpenAI-compatible API at `http://<PI_HOST>:8080/v1/chat/completions`

**Access from Windows:**
- SSH: `ssh <PI_USER>@<PI_HOST>`
- API: `http://<PI_HOST>:8080/v1/chat/completions`
- Health: `ssh <PI_TARGET> "~/korgi/server/health_check.sh"`
- Logs: `ssh <PI_TARGET> "sudo journalctl -u korgi-phi3 -f"`

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| OOM kill during inference | Context too large | Reduce `-c` from 2048 to 1024 in `start_phi3.sh` |
| Very slow generation (<2 tok/s) | Swap thrashing | Check `free -h`; kill memory-hungry processes; reduce context |
| Garbled output | Model file corrupted | Re-verify SHA256; re-transfer if mismatch |
| Port 8080 unreachable from Windows | Firewall or wrong IP | `sudo ufw allow 8080/tcp`; verify Pi IP with `hostname -I` |
| Service won't start | Path wrong in service file | Check `journalctl -u korgi-phi3 -e`; verify paths match actual username |
| `hailo-ollama` conflicts | Port collision | They use different ports (8000 vs 8080); no conflict expected |
| Build fails on Pi | Missing deps | Re-run `sudo apt install build-essential cmake git libcurl4-openssl-dev` |
