from collections import defaultdict
import time

def profile(f):
    """ Convenience function to make decorator tags simpler:
        e.g. @profile instead of @Profiler.profile """
    return Profiler.profile(f)

class Profiler:
    """ A code profiling class. Keeps track of function calls and running time. """
    calls = defaultdict(int)  # default = 0
    time = defaultdict(float)  # default = 0.0
    start_time = None
    time_limit = None

    @staticmethod
    def _add(function_name, sec):
        """ Add 1 call and <sec> time to named function tracking """
        Profiler.calls[function_name] += 1
        Profiler.time[function_name] += sec

    @staticmethod
    def profile(f):
        """ The profiling decorator """
        def wrapper(*args, **kwargs):
            if Profiler.time_limit and time.time() - Profiler.start_time > Profiler.time_limit:
                raise TimeoutError(f"Time limit of {Profiler.time_limit} seconds reached.")
            function_name = f.__name__  # Use f.__name__ to get the function name
            start = time.time_ns()
            val = f(*args, **kwargs)
            elapsed_time_sec = (time.time_ns() - start) / 10**9  # Convert ns to seconds
            Profiler._add(function_name, elapsed_time_sec)
            return val
        return wrapper

    @staticmethod
    def set_time_limit(seconds):
        """ Set the time limit for profiling """
        Profiler.time_limit = seconds
        Profiler.start_time = time.time()

    @staticmethod
    def report(filename=None):
        """ Summarize # calls, total runtime, and time/call for each function """
        report_data = "Function              Calls     TotSec   Sec/Call\n"
        for name, num in Profiler.calls.items():
            sec = Profiler.time[name]
            avg_time = sec / num if num > 0 else 0
            report_data += f'{name:20s} {num:6d} {sec:10.6f} {avg_time:10.6f}\n'

        if filename:
            with open(filename, 'w') as f:
                f.write(report_data)
        else:
            print(report_data)