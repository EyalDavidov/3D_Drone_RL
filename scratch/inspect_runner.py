from rsl_rl.runners import OnPolicyRunner
import inspect
# Let's inspect the logger attribute or class in rsl_rl
from rsl_rl.utils import logger
try:
    print(inspect.getsource(logger.Logger))
except Exception as e:
    print(e)
