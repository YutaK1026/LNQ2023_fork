from collections import deque
import torch
from preregistration.imi_varreg.imi_debug_tools import debug_print


class StopCriterion:
    def __init__(self, use_increase_count=True, maximum_increase_count=10, line_regression_mode='standard', fitting_iterations=20,
                 regression_line_slope_threshold=0.0005):
        self.current_increase_count = 0
        self.maximum_increase_count = maximum_increase_count
        self.fitting_array = deque(maxlen=fitting_iterations)
        self.regression_line_slope_threshold = regression_line_slope_threshold
        self.line_regression_mode = line_regression_mode
        self.min_value = 1e10
        self.max_value = -1e10
        self.increase_count = 0
        self.perform_line_fitting = (line_regression_mode != 'none')
        self.perform_increase_count = use_increase_count

    def append(self, value):
        self.increase_count = 0 if value <= self.min_value else self.increase_count + 1
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.fitting_array.append(value)

    def check_stop_criterion(self):
        if self.perform_increase_count and (self.increase_count > self.maximum_increase_count):
            debug_print(3, f"Stop registration because increase count reached ! ({self.increase_count})")
            return True
        if len(self.fitting_array) < self.fitting_array.maxlen:
            return False
        if self.line_regression_mode == 'none' or not self.perform_line_fitting:
            return False
        elif self.line_regression_mode == 'standard':
            x = torch.tensor(list(range(len(self.fitting_array))))
            y = torch.tensor(list(self.fitting_array))
        elif self.line_regression_mode == 'scaled':
            x = torch.tensor(list(range(len(self.fitting_array))))
            y = torch.tensor(list(self.fitting_array)) / self.max_value
        elif self.line_regression_mode == 'normalized':
            x = torch.tensor(list(range(len(self.fitting_array))))
            y = (torch.tensor(list(self.fitting_array)) - self.min_value) / (self.max_value - self.min_value)
        else:
            raise ValueError(f"Unknown line fitting mode '{self.line_regression_mode}', support are 'none'|'standard'|'scaled'|'normalized'")

        m, b = self.linear_fit(x, y)
        debug_print(5, f"Fitted line (m * x + b): m={m}, b={b}")
        if m > 0:
            debug_print(3, f"Stop registration because of increasing regression line ! (m={m})")
            return True
        if m.abs() < self.regression_line_slope_threshold:
            debug_print(3, f"Stop registration because slope of regression line is below threshold ! (m={m})")
            return True
        return False

    @staticmethod
    def linear_fit(x, y):
        assert len(x) == len(y)
        n = len(x)
        if n <= 1:
            return 0, 0
        divisor = x.pow(2).sum() - x.sum()**2 / n
        if divisor.abs() < 1e-8:
            return 0, 0
        m = ((x * y).sum() - (x.sum() * y.sum() / n)) / divisor
        b = (y.sum() - m * x.sum()) / n
        return m, b
