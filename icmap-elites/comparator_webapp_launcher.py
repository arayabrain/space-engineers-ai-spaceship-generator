import argparse
from pcgsepy.guis.initial_ships_eval.webapp import app, set_app_layout

parser = argparse.ArgumentParser()
# parser.add_argument("--mapelites_file", help="Location of the MAP-Elites object",
#                     type=str, default=None)
parser.add_argument("--debug", help="Launch the webapp in debug mode",
                    action='store_true')
parser.add_argument("--host", help="Specify host address",
                    type=str, default='127.0.0.1')
parser.add_argument("--port", help="Specify port",
                    type=int, default=8050)
parser.add_argument("--use_reloader", help="Use reloader (set to True when using Jupyter Notebooks)",
                    action='store_false')

args = parser.parse_args()

set_app_layout(spaceships=[
    "cockpit(1)corridorsimple(1)thrusters(1)",
    "cockpit(1)corridorsimple(1)corridorcargo(1)thrusters(1)",
    "cockpit(1)corridorsimple(1)[RotYccwXcorridorsimple(1)]thrusters(1)",
    "cockpit(1)corridorsimple(1)[RotYcwXcorridorsimple(1)thrusters]thrusters(1)",
])

app.run_server(debug=args.debug,
               host=args.host,
               port=args.port,
               use_reloader=args.use_reloader)