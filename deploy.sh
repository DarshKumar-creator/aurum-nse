#!/bin/bash
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; GOLD='\033[0;33m'; NC='\033[0m'

echo -e "${GOLD}⚡ AURUM v18 — Deploy${NC}"
echo ""

# Check git is configured
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo -e "${RED}✗ Not a git repo. Run:${NC}"
  echo "  git init && git remote add origin https://github.com/DarshKumar-creator/aurum-nse.git"
  exit 1
fi

# Check remote
REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE" ]; then
  echo -e "${RED}✗ No remote. Run:${NC}"
  echo "  git remote add origin https://github.com/DarshKumar-creator/aurum-nse.git"
  exit 1
fi
echo -e "  Remote: ${GREEN}$REMOTE${NC}"

# Stage all
git add -A
CHANGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
if [ "$CHANGED" = "0" ]; then
  echo "  Nothing changed since last push."
else
  echo "  Staging $CHANGED file(s)"
fi

# Commit
MSG="${1:-v18 fix startup lifespan + connection + TD API $(date '+%Y-%m-%d %H:%M')}"
git commit -m "$MSG" 2>/dev/null || echo "  (nothing new to commit)"

# Push
echo -e "  Pushing to ${GREEN}main${NC}..."
git push origin main

echo ""
echo -e "${GREEN}✅ Pushed. Render will auto-deploy in ~2 min.${NC}"
echo ""
echo "  Watch build:  https://dashboard.render.com"
echo "  Health check: https://aurum-nse.onrender.com/ping"
echo "  Live app:     https://aurum-nse.onrender.com"
echo ""
echo "  After deploy, first page load may take 30-60s (cold start)."
echo "  Status bar will show which URL it's connecting to."
echo "  Open DevTools console to see [detect] logs."
