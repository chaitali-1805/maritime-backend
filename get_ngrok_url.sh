#!/bin/bash

# Get ngrok public URL
# Usage: ./get_ngrok_url.sh

echo "========================================="
echo "Getting ngrok public URL..."
echo "========================================="

# Check if ngrok container is running
if ! docker ps | grep -q ngrok-tunnel; then
    echo "❌ Error: ngrok container is not running"
    echo ""
    echo "Start it with: docker-compose up -d ngrok"
    exit 1
fi

# Method 1: Using ngrok API
echo ""
echo "Method 1: From ngrok API"
URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url' 2>/dev/null)

if [ -n "$URL" ] && [ "$URL" != "null" ]; then
    echo "✅ Public URL: $URL"
    echo ""
    echo "Test it:"
    echo "  curl $URL/health"
    echo ""
    echo "Use in Vercel frontend:"
    echo "  NEXT_PUBLIC_API_URL=$URL"
    echo ""
    
    # Test the URL
    echo "Testing connection..."
    if curl -s -f "$URL/health" > /dev/null 2>&1; then
        echo "✅ Backend is accessible!"
    else
        echo "⚠️  Warning: Could not connect to backend"
        echo "   Make sure backend container is running"
    fi
else
    echo "❌ Could not get URL from API"
    echo ""
    echo "Method 2: Check logs"
    docker-compose logs ngrok | grep -i "url=" | tail -1
fi

echo ""
echo "========================================="
echo "ngrok Web Interface: http://localhost:4040"
echo "========================================="