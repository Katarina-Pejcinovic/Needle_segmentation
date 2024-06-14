import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

#measurements 
Dice = [0.29362,0.26197, 0.39792] 
recall = [0.619, 0.686, 0.604]
Dice = np.array(Dice)
recall = np.array(recall)


# Mean to compare against
comparison_mean = 0.3

# Perform Wilcoxon signed-rank test
recall_stat, recall_p = wilcoxon(recall - comparison_mean, alternative='greater')
dice_stat, dice_p = wilcoxon(Dice - comparison_mean, alternative='greater')

print(f'recall: p-value={recall_p}')
print(f'DICE:  p-value={dice_p}')
