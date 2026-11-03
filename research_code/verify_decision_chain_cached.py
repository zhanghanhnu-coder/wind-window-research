"""Identical independent audit with the proven result-readback compatibility layer."""
from highs_readback_compat import install
install()
from verify_decision_chain import run
if __name__=='__main__':run()
