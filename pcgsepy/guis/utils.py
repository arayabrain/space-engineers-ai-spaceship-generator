from datetime import datetime
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pcgsepy.config import BIN_POP_SIZE, CS_MAX_AGE, MY_EMITTERS
from pcgsepy.mapelites.behaviors import (BehaviorCharacterization, avg_ma,
                                         mame, mami, symmetry)

from pcgsepy.mapelites.map import MAPElites


class Metric:
    def __init__(self,
                 emitters: List[str],
                 exp_n: int,
                 name: str,
                 multiple_values: bool = False) -> None:
        """Create a `Metric` object.

        Args:
            emitters (List[str]): The list of emitters names.
            exp_n (int): The current experiment number.
            name (str): The name of the metric.
            multiple_values (bool, optional): Whether this metric tracks multiple values per generation or a sum. Defaults to False.
        """
        self.name = name
        self.current_generation: int = 0
        self.multiple_values = multiple_values
        self.history: Dict[int, List[Any]] = {
            self.current_generation: [] if multiple_values else 0
        }
        self.emitter_names: List[str] = [emitters[exp_n]]
    
    def add(self,
            value: Any):
        """Add a new value to the current generation.

        Args:
            value (Any): The new value.
        """
        if self.multiple_values:
            self.history[self.current_generation].append(value)
        else:
            self.history[self.current_generation] += value
    
    def reset(self):
        """Clear the metric trackings."""
        if self.multiple_values:
            self.history[self.current_generation] = []
        else:
            self.history[self.current_generation] = 0
    
    def new_generation(self,
                       emitters: List[str],
                       exp_n: int):
        """Start a new generation to track.

        Args:
            emitters (List[str]): The list of emitter names.
            exp_n (int): The current experiment number.
        """
        self.current_generation += 1
        self.reset()
        self.emitter_names.append(emitters[exp_n])
    
    def get_averages(self) -> List[Any]:
        """Get the metric averages over the history.

        Returns:
            List[Any]: The list of averages.
        """
        return [np.mean(l) for l in self.history.values()]


class Semaphore:
    def __init__(self,
                 locked: bool = False) -> None:
        """Create a `Semamphore` object.

        Args:
            locked (bool, optional): Initial locked value. Defaults to False.
        """
        self._is_locked = locked
        self._running = ''
    
    @property
    def is_locked(self) -> bool:
        """Check if the semaphore is currently locked.

        Returns:
            bool: The locked value.
        """
        return self._is_locked
    
    def lock(self,
             name: Optional[str] = '') -> None:
        """Lock the semaphore.

        Args:
            name (Optional[str], optional): The locking process name. Defaults to ''.
        """
        self._is_locked = True
        self._running = name
    
    def unlock(self) -> None:
        """Unlock the semaphore"""
        self._is_locked = False
        self._running = ''


class DashLoggerHandler(logging.StreamHandler):
    def __init__(self):
        """Create a new logging handler.
        """
        logging.StreamHandler.__init__(self)
        self.queue = []

    def emit(self,
             record: Any) -> None:
        """Process the incoming record.

        Args:
            record (Any): The new logging record.
        """
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = self.format(record)
        self.queue.append(f'[{t}]\t{msg}')


class AppMode(Enum):
    """Enumerator for the application mode."""
    USERSTUDY = 0
    USER = 1
    DEV = 2


class AppSettings:
    def __init__(self) -> None:
        """Generate a new `AppSettings` object."""
        self.current_mapelites: Optional[MAPElites] = None
        self.exp_n: int = 0
        self.gen_counter: int = 0
        self.hm_callback_props: Dict[str, Any] = {}
        self.my_emitterslist: List[str] = MY_EMITTERS.copy()
        self.behavior_descriptors: List[BehaviorCharacterization] = [
            BehaviorCharacterization(name='Major axis / Medium axis',
                                    func=mame,
                                    bounds=(0, 10)),
            BehaviorCharacterization(name='Major axis / Smallest axis',
                                    func=mami,
                                    bounds=(0, 20)),
            BehaviorCharacterization(name='Average Proportions',
                                    func=avg_ma,
                                    bounds=(0, 20)),
            BehaviorCharacterization(name='Symmetry',
                                    func=symmetry,
                                    bounds=(0, 1))
        ]
        self.rngseed: int = None
        self.selected_bins: List[Tuple[int, int]] = []
        self.step_progress: int = -1
        self.use_custom_colors: bool = True
        self.app_mode: AppMode = None
        self.consent_ok: bool = None

    def initialize(self,
                   mapelites: MAPElites,
                   dev_mode: bool = False):
        """Initialize the object.

        Args:
            mapelites (MAPElites): The MAP-Elites object.
            dev_mode (bool, optional): Whether to set the application to developer mode. Defaults to False.
        """
        self.current_mapelites = mapelites
        self.app_mode = AppMode.DEV if dev_mode else self.app_mode
        self.hm_callback_props['pop'] = {
            'Feasible': 'feasible',
            'Infeasible': 'infeasible'
        }
        self.hm_callback_props['metric'] = {
            'Fitness': {
                'name': 'fitness',
                'zmax': {
                    'feasible': sum([x.weight * x.bounds[1] for x in self.current_mapelites.feasible_fitnesses]) + self.current_mapelites.nsc,
                    'infeasible': 1.
                },
                'colorscale': 'Inferno'
            },
            'Age':  {
                'name': 'age',
                'zmax': {
                    'feasible': CS_MAX_AGE,
                    'infeasible': CS_MAX_AGE
                },
                'colorscale': 'Greys'
            },
            'Coverage': {
                'name': 'size',
                'zmax': {
                    'feasible': BIN_POP_SIZE,
                    'infeasible': BIN_POP_SIZE
                },
                'colorscale': 'Hot'
            }
        }
        self.hm_callback_props['method'] = {
            'Population': True,
            'Elite': False
        }