import time
from functools import wraps

def cooldown(seconds):
    def decorator(func):
        last_called = [0]  # This will hold the last call time for each function

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            if current_time - last_called[0] < seconds:
                return None
            
            last_called[0] = current_time
            return func(*args, **kwargs)

        return wrapper
    return decorator
