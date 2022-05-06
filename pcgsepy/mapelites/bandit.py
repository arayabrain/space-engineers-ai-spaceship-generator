from typing import List, Optional
import numpy as np


class Bandit:
    def __init__(self,
                 action: str):
        """Bandit class.

        Args:
            action (str): The action that is applied by the bandit.
        """
        self.action = action
        self.tot_rewards = 0.
        self.tot_actions = 0
    
    def __str__(self) -> str:
        return f'{self.action}_{str(self.avg_rewards)}'
    
    @property
    def avg_rewards(self) -> float:
        """Get the average reward of the bandit.

        Returns:
            float: The average reward.
        """
        return 0 if self.tot_actions == 0 else self.tot_rewards / self.tot_actions


class EpsilonGreedyAgent:
    def __init__(self,
                 bandits: List[Bandit],
                 epsilon: Optional[float]) -> None:
        """Simulate an epsilon-greedy multiarmed bandit agent.

        Args:
            bandits (List[Bandit]): The bandits.
            epsilon (Optional[float]): A fixed epsilon value. If not set, a decayed epsilon will be used.
        """
        assert len(bandits) > 0, 'Can\'t initialize an agent without bandits!'
        self.bandits = bandits
        self.epsilon = epsilon
        self.tot_actions = 0
    
    def __str__(self) -> str:
        return f'{len(self.bandits)}-armed bandit ε-greedy agent'
    
    def _get_random_bandit(self) -> Bandit:
        """Get a random bandit from the available bandits.

        Returns:
            Bandit: A randomly picked bandit.
        """
        return np.random.choice(self.bandits)
    
    def _get_best_bandit(self) -> Bandit:
        """Get the best (highest reward) bandit.

        Returns:
            Bandit: The bandit with the highest reward.
        """
        return self.bandits[np.argmax([x.avg_rewards for x in self.bandits])]
    
    def choose_bandit(self) -> Bandit:
        """Pick a bandit. Applies epsilon-greedy policy when picking.

        Returns:
            Bandit: The selected bandit.
        """
        p = np.random.uniform(low=0, high=1, size=1) < (self.epsilon or 1 / (1 + self.tot_actions))
        return self._get_random_bandit() if p else self._get_best_bandit()
    
    def reward_bandit(self,
                      bandit: Bandit,
                      reward: float) -> None:
        self.tot_actions += 1
        bandit.tot_actions += 1
        bandit.tot_rewards += reward