import argparse
from pcgsepy.guis.ships_comparator.webapp import app, set_app_layout

parser = argparse.ArgumentParser()
parser.add_argument("--debug", help="Launch the webapp in debug mode",
                    action='store_true')
parser.add_argument("--host", help="Specify host address",
                    type=str, default='127.0.0.1')
parser.add_argument("--port", help="Specify port",
                    type=int, default=8050)
parser.add_argument("--use_reloader", help="Use reloader (set to True when using Jupyter Notebooks)",
                    action='store_false')

args = parser.parse_args()

set_app_layout()

app.run_server(debug=args.debug,
               host=args.host,
               port=args.port,
               use_reloader=args.use_reloader)