#!/bin/bash

echo "🚀 Starting SAR Detection API..."

# Start backend
docker compose up -d sar-backend

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 15

# Remove old tunnel if exists
docker rm -f cloudflared-tunnel 2>/dev/null

# Start Cloudflare quick tunnel
echo "🌐 Creating Cloudflare Tunnel..."
docker run -d \
  --name cloudflared-tunnel \
  --network host \
  cloudflare/cloudflared:latest \
  tunnel --url http://localhost:8000

# Wait for tunnel to connect
sleep 10

# Extract and display URL
TUNNEL_URL=$(docker logs cloudflared-tunnel 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)

echo ""
echo "================================"
echo "✅ API is LIVE at:"
echo "$TUNNEL_URL"
echo "================================"
echo ""
echo "📝 URL saved to: tunnel-url.txt"
echo ""

# Save URL to file
echo "$TUNNEL_URL" > tunnel-url.txt
echo "Last updated: $(date)" >> tunnel-url.txt

# Test the API
echo "🧪 Testing API health check..."
sleep 5
curl -s "$TUNNEL_URL/health" | jq . || echo "API is starting up..."

echo ""
echo "📊 View logs:"
echo "  Backend: docker logs -f sar-backend"
echo "  Tunnel:  docker logs -f cloudflared-tunnel"
echo ""
echo "🛑 To stop: docker compose down && docker rm -f cloudflared-tunnel"