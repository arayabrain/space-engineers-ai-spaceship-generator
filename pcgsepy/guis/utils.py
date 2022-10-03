from datetime import datetime
from enum import Enum, auto
import logging
from typing import Any, Dict, List, Optional

import numpy as np


class Metric:
    def __init__(self,
                 emitters: List[str],
                 exp_n: int,
                 multiple_values: bool = False) -> None:
        self.current_generation: int = 0
        self.multiple_values = multiple_values
        self.history: Dict[int, List[Any]] = {
            self.current_generation: [] if multiple_values else 0
        }
        self.emitter_names: List[str] = [emitters[exp_n]]
    
    def add(self,
            value: Any):
        if self.multiple_values:
            self.history[self.current_generation].append(value)
        else:
            self.history[self.current_generation] += value
    
    def reset(self):
        if self.multiple_values:
            self.history[self.current_generation] = []
        else:
            self.history[self.current_generation] = 0
    
    def new_generation(self,
                       emitters: List[str],
                       exp_n: int):
        self.current_generation += 1
        self.reset()
        self.emitter_names.append(emitters[exp_n])
    
    def get_averages(self) -> List[Any]:
        return [np.mean(l) for l in self.history.values()]


class Semaphore:
    def __init__(self,
                 locked: bool = False) -> None:
        self._is_locked = locked
        self._running = ''
    
    @property
    def is_locked(self) -> bool:
        return self._is_locked
    
    def lock(self,
             name: Optional[str] = ''):
        self._is_locked = True
        self._running = name
    
    def unlock(self):
        self._is_locked = False
        self._running = ''


class DashLoggerHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        self.queue = []

    def emit(self, record):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = self.format(record)
        self.queue.append(f'[{t}]\t{msg}')


class AppMode(Enum):
    USERSTUDY = 0
    USER = 1
    DEV = 2