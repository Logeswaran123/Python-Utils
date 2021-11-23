import time

class Timer:
    def __init__(self):

        self._current_time = 0.0
        self._total_time = 0.0
        self._counter = 0.0

    def start(self, start_time=None):
        if start_time is not None:
            assert type(start_time) in [int, float]
            self._current_time = start_time
        else:
            self._current_time = time.time()

    def end(self, end_time=None):
        if end_time is not None:
            assert type(end_time) in [int, float]
            self._current_time = end_time - self._current_time
            self._total_time += self._current_time
        else:
            self._current_time = time.time() - self._current_time
            self._total_time += self._current_time
            self._counter += 1

    def reset(self):
        self._current_time = 0.0
        self._total_time = 0.0
        self._counter = 0.0

    def get_current_time(self):
        return self._current_time

    def get_average_time(self):
        avg_time = 0
        try:
            avg_time = self._total_time/self._counter
            return avg_time
        except ZeroDivisionError:
            return 0

    def get_current_fps(self):
        fps = 0
        try:
            fps = 1/self._current_time
            return fps
        except ZeroDivisionError:
            return 0

    def get_average_fps(self):
        avg_fps = 0
        try:
            avg_fps = 1/(self._total_time/self._counter)
            return avg_fps
        except ZeroDivisionError:
            return 0
