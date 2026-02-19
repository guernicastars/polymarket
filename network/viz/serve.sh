#!/bin/bash
# Serve the Polymarket Signals unified visualization platform
#
# Usage: cd network/viz && bash serve.sh
# Or:    bash serve.sh 3000    (custom port)

PORT=${1:-8888}
echo ""
echo "  Polymarket Signals — Network Intelligence Platform"
echo "  ==================================================="
echo ""
echo "  Main site:  http://localhost:$PORT"
echo ""
echo "  Individual pages also accessible:"
echo "    Donbas:       http://localhost:$PORT/donbas-network.html"
echo "    Ukraine:      http://localhost:$PORT/ukraine-network.html"
echo "    Middle East:  http://localhost:$PORT/mideast-network.html"
echo "    Market Graph: http://localhost:$PORT/market-graph.html"
echo "    Dual Mode:    http://localhost:$PORT/dual-mode.html"
echo "    Showcase:     http://localhost:$PORT/showcase.html"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

python3 -m http.server "$PORT"
