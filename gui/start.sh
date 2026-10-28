#!/bin/bash
# opens the native window (app.py, ~170 MB of RAM); --browser opens the web page instead (~1 GB in Firefox)
cd "$(dirname "$0")"
mkdir -p data
if [ "$1" != --browser ] && python3 -c 'import gi; gi.require_version("Gtk", "4.0"); gi.require_version("Adw", "1")' 2>/dev/null; then
  exec ./app.py
fi
URL=http://127.0.0.1:7860
curl -s -o /dev/null $URL/api/config || { nohup ./server.py > data/server.log 2>&1 & sleep 1; }
xdg-open $URL >/dev/null 2>&1 &
