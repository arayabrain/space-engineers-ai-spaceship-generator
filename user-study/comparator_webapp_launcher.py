import logging
import os
import sys

from waitress import serve

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    os.chdir(sys._MEIPASS)

import argparse
import webbrowser
from pcgsepy.guis.ships_comparator.webapp import app, set_app_layout

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="Launch the webapp in debug mode",
                    action='store_true')
parser.add_argument("--host", help="Specify host address",
                    type=str, default='127.0.0.1')
parser.add_argument("--port", help="Specify port",
                    type=int, default=8051)
parser.add_argument("--use_reloader", help="Use reloader (set to True when using Jupyter Notebooks)",
                    action='store_false')

args = parser.parse_args()

logging.getLogger('werkzeug').setLevel(logging.ERROR)

set_app_layout()

webapp_url = f'http://{args.host}:{args.port}/'
print(f'Serving webapp on http://{args.host}:{args.port}/...')
webbrowser.open_new(webapp_url)

# close the splash screen if launched via application
try:
    import pyi_splash
    if pyi_splash.is_alive():
        pyi_splash.close()
except ModuleNotFoundError as e:
    pass

serve(app.server,
      threads=16,
      host=args.host,
      port=args.port)
