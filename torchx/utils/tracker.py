import time
from contextlib import contextmanager
from threading import Thread, Lock


class SimpleTimer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('elapsed', self.interval)


class MovingAverageRecorder:
    """
        Records moving average
    """
    def __init__(self, decay=0.95):
        self.decay = decay
        self.cum_value = 0
        self.normalization = 0

    def add_value(self, value):
        self.cum_value *= self.decay
        self.cum_value += value

        self.normalization *= self.decay
        self.normalization += 1

        return self.cum_value / self.normalization

    def cur_value(self):
        """
            Returns current moving average, 0 if no records
        """
        if self.normalization == 0:
            return 0
        else:
            return self.cum_value / self.normalization


class ThreadSafeMovingAverageRecorder(MovingAverageRecorder):
    def __init__(self, decay=0.95):
        super().__init__()
        self.lock = Lock()

    def add_value(self, value):
        with self.lock:
            return super().add_value(value)

    def cur_value(self):
        with self.lock:
            return super().cur_value()


class TimeRecorder:
    """
        Records average of whatever context block it is recording
        Don't call time in two threads
    """
    def __init__(self, decay=0.9995, max_seconds=10):
        """
        Args:
            decay: Decay factor of smoothed moving average
                    Default is 0.9995, which is approximately moving average
                    of 2000 samples
            max_seconds: round down all time differences larger than specified
                        Useful when the application just started and there are long waits
                        that might throw off the average
        """
        self.moving_average = ThreadSafeMovingAverageRecorder(decay)
        self.max_seconds = max_seconds
        self.started = False

    @contextmanager
    def time(self):
        pre_time = time.time()
        yield None
        post_time = time.time()

        interval = min(self.max_seconds, post_time - pre_time)
        self.moving_average.add_value(interval)

    def start(self):
        if self.started:
            raise RuntimeError('Starting a started timer')
        self.pre_time = time.time()
        self.started = True

    def stop(self):
        if not self.started:
            raise RuntimeError('Stopping a timer that is not started')
        self.post_time = time.time()
        self.started = False

        interval = min(self.max_seconds, self.post_time - self.pre_time)
        self.moving_average.add_value(interval)

    def lap(self):
        if not self.started:
            raise RuntimeError('Stopping a timer that is not started')
        post_time = time.time()

        interval = min(self.max_seconds, post_time - self.pre_time)
        self.moving_average.add_value(interval)

        self.pre_time = post_time

    @property
    def avg(self):
        return self.moving_average.cur_value()


class PeriodicWakeUpWorker(Thread):
    """
    Args:
        @target: The function to be called periodically
        @interval: Time between two calls
        @args: Args to feed to target()
        @kwargs: Key word Args to feed to target()
    """
    def __init__(self, target, interval=1, args=None, kwargs=None):
        Thread.__init__(self)
        self.target = target
        self.interval = interval
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.args is None:
            self.args = []
        if self.kwargs is None:
            self.kwargs = {}
        while True:
            self.target(*self.args, **self.kwargs)
            time.sleep(self.interval)


class TimedTracker(object):
    def __init__(self, interval):
        self.init_time = time.time()
        self.last_time = self.init_time
        self.interval = interval

    def track_increment(self):
        cur_time = time.time()
        time_since_last = cur_time - self.last_time
        enough_time_passed = time_since_last >= self.interval
        if enough_time_passed:
            self.last_time = cur_time
        return enough_time_passed


class AverageValue(object):
    """
        Keeps track of average of things
        Always caches the latest value so no division by 0
    """
    def __init__(self, initial_value):
        self.last_val = initial_value
        self.sum = initial_value
        self.count = 1

    def add(self, value):
        self.last_val = value
        self.sum += value
        self.count += 1

    def avg(self, clear=True):
        """
            Get the average of the currently tracked value
        Args:
            @clear: if true (default), clears the cached sum/count
        """
        ans = self.sum / self.count
        if clear:
            self.sum = self.last_val
            self.count = 1
        return ans


class AverageDictionary(object):
    def __init__(self):
        self.data = {}

    def add_scalars(self, new_data):
        for key in new_data:
            if key in self.data:
                self.data[key].add(new_data[key])
            else:
                self.data[key] = AverageValue(new_data[key])

    def get_values(self, clear=True):
        response = {}
        for key in self.data:
            response[key] = self.data[key].avg(clear=clear)
        return response

