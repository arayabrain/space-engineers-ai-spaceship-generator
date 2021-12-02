from .rules import StochasticRules


class LParser:
    def __init__(self,
                 rules: StochasticRules):
        self.rules = rules

    def expand(self,
               axiom: str) -> str:
        i = 0
        while i < len(axiom):
            for k in self.rules.get_lhs():
                if axiom[i:].startswith(k):
                    rhs = self.rules.get_rhs(k)
                    axiom = axiom[:i] + rhs + axiom[i + len(k):]
                    # since we use multiple characters in LHS, we need to skip them in the current iteration
                    i += len(rhs) - 1
                    break
            i += 1
        return axiom
