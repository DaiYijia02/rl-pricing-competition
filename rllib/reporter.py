from ray.tune import ProgressReporter

class CustomReporter(ProgressReporter):

    def should_report(self, trials, done=False):
        return True

    def report(self, trials, *sys_info):
        print(*sys_info)
        print("\n".join([str(trial) for trial in trials]))