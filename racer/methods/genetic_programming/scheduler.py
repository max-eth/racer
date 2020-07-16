class Scheduler:
    def __init__(self, *, milestones, **schedule):
        self.step_count = 0
        self.schedule = schedule
        self.milestones = milestones

    def step(self):
        self.step_count += 1

    def get(self, param_name):
        param_schedule = self.schedule[param_name]
        assert len(param_schedule) == len(self.milestones)
        i = 0
        while self.step_count >= self.milestones[i] and i < len(param_schedule):
            val = param_schedule[i]
            i += 1
        return val
