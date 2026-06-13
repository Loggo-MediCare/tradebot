import os
import builtins

def should_print_roi():
    v = os.environ.get('NO_ROI_OUTPUT')
    if v is None:
        return True
    return str(v).lower() not in ('1', 'true', 'yes')

def print_roi(*args, **kwargs):
    if should_print_roi():
        builtins.print(*args, **kwargs)

def silent_print(*args, **kwargs):
    # alias for tests or explicit silencing
    return None
