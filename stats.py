import numpy as np
from scipy import stats

# Your three measurements
mean1 = 0.296*0.29362 + 0.704*0.619
mean2 = 0.296*0.26197 + 0.704*0.686
mean3 = 0.296*0.39792 + 0.704*0.604
measurements = [mean1, mean2, mean3]  # Replace mean1, mean2, and mean3 with your actual measurements

# Mean to compare against
comparison_mean = 0.296

# Perform one-sample one-sided t-test
t_statistic, p_value = stats.ttest_1samp(measurements, comparison_mean, alternative= 'greater')


print(f"one-sided p-value: {p_value}")
