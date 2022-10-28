import PyInstaller.__main__ as pyinst
from datetime import date

from pcgsepy.config import USE_TORCH


app_name = f'Spaceships Comparator ({"with" if USE_TORCH else "no"} Pytorch)_{date.today().strftime("%Y%m%d")}'

pyi_args = ['comparator_webapp_launcher.py',
            '--clean',
            '--onefile',
            '--noconfirm',
            '--name', f"{app_name}",
            '--icon', 'assets\\favicon.ico',
            '--splash', 'assets\\thumb.png',
            '--add-data', './estimators;estimators',
            '--add-data', './assets;assets',
            '--add-data', './block_definitions.json;.',
            '--add-data', './common_atoms.json;.',
            '--add-data', './configs.ini;.',
            '--add-data', './hl_atoms.json;.',
            '--add-data', './hlrules;.',
            '--add-data', './hlrules_sm;.',
            '--add-data', './llrules;.']


pyinst.run(pyi_args)
