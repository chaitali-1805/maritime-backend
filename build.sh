#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   SAR Ship Detection - Build & Deploy Script   ${NC}"
echo -e "${GREEN}==================================================${NC}"

# Step 1: Stop and remove existing containers
echo -e "\n${YELLOW}[1/5] Stopping existing containers...${NC}"
docker compose down -v 2>/dev/null || true

# Step 2: Clean up Docker cache (optional but recommended)
echo -e "\n${YELLOW}[2/5] Cleaning Docker cache...${NC}"
docker system prune -f

# Step 3: Build with no cache to ensure fresh build
echo -e "\n${YELLOW}[3/5] Building Docker image (this may take several minutes)...${NC}"
docker compose build --no-cache

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed! Please check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful!${NC}"

# Step 4: Start the container
echo -e "\n${YELLOW}[4/5] Starting container...${NC}"
docker compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start container!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Container started!${NC}"

# Step 5: Wait for service to be healthy and show logs
echo -e "\n${YELLOW}[5/5] Waiting for service to start (60s timeout)...${NC}"
echo -e "${YELLOW}Showing live logs (press Ctrl+C to stop watching logs):${NC}\n"

# Show logs for 60 seconds or until service is ready
timeout 60 docker compose logs -f &
LOGS_PID=$!

# Wait for health check
for i in {1..30}; do
    sleep 2
    if docker compose ps | grep -q "healthy"; then
        kill $LOGS_PID 2>/dev/null
        echo -e "\n${GREEN}✅ Service is healthy and ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        kill $LOGS_PID 2>/dev/null
        echo -e "\n${YELLOW}⚠️  Service started but health check not confirmed yet${NC}"
        echo -e "${YELLOW}Check logs with: docker compose logs -f${NC}"
    fi
done

# Show service info
echo -e "\n${GREEN}==================================================${NC}"
echo -e "${GREEN}   Service Information${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
echo -e "Health:    ${GREEN}http://localhost:8000/health${NC}"
echo -e "\n${YELLOW}Useful Commands:${NC}"
echo -e "  View logs:        ${GREEN}docker compose logs -f${NC}"
echo -e "  Stop service:     ${GREEN}docker compose down${NC}"
echo -e "  Restart service:  ${GREEN}docker compose restart${NC}"
echo -e "  Check status:     ${GREEN}docker compose ps${NC}"
echo -e "${GREEN}==================================================${NC}\n"

# Test health endpoint
echo -e "${YELLOW}Testing health endpoint...${NC}"
sleep 5
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health endpoint responding!${NC}\n"
else
    echo -e "${YELLOW}⚠️  Health endpoint not responding yet, check logs${NC}\n"
fi