from typing import List
import numpy as np


class StochasticRules:
    def __init__(self):
        self._rules = {}
        self.lhs_alphabet = set()

    def add_rule(self,
                 lhs: str,
                 rhs: str,
                 p: float) -> None:
        if lhs in self._rules.keys():
            self._rules[lhs][0].append(rhs)
            self._rules[lhs][1].append(p)
        else:
            self._rules[lhs] = ([rhs], [p])
        lhs = lhs.replace('(x)', '').replace(']', '')
        self.lhs_alphabet.add(lhs)

    def rem_rule(self,
                 lhs: str) -> None:
        self._rules.pop(lhs)
        lhs = lhs.replace('(x)', '').replace(']', '')
        self.lhs_alphabet.pop(lhs)

    def get_lhs(self) -> List[str]:
        return self._rules.keys()

    def get_rhs(self,
                lhs: str) -> str:
        rhs, p = self._rules[lhs]
        return np.random.choice(rhs, p=p)

    def validate(self):
        for lhs in self._rules.keys():
            p = sum(self._rules[lhs][1])
            assert np.isclose(p, 1.), f'Probability must sum to 1: found {p} for `{lhs}`.'
    
    def __str__(self) -> str:
        s = []
        for k in self._rules.keys():
            for o, p in zip(*self._rules[k]):
                s.append(f'{k} {p} {o}')
        return '\n'.join(s)


class RuleMaker:
    def __init__(self,
                 ruleset: str):
        with open(ruleset, 'r') as f:
            self.ruleset = f.readlines()

    def get_rules(self) -> StochasticRules:
        rules = StochasticRules()
        for rule in self.ruleset:
            if rule.startswith('#'):  # comment in configuration file
                pass
            else:
                lhs, p, rhs = rule.strip().split(' ')
                rules.add_rule(lhs=lhs,
                               rhs=rhs,
                               p=float(p))
        rules.validate()
        return rules