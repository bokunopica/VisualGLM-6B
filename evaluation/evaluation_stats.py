import numpy as np
import scipy.stats as st
# data of goals scored by 20 footballers in a calendar year

fb_data = [0.3952109070946664, 0.3836508466813337, 0.3831080071626834, 0.39720851900748794, 0.3853174429630886, 0.39312517266021174, 0.3880164162084076, 0.391110124931037, 0.3939831058746467, 0.39794207454787717]

# 置信水平create 95% confidence interval
confidence_level = 0.95
# t.interval() 计算置信区间 （n<30）
print("置信区间为:",st.t.interval(confidence_level, df=len(fb_data)-1,
              loc=np.mean(fb_data),
              scale=st.sem(fb_data)))


# fb_data = np.random.randint(15, 20, 80)
# create 90% confidence interval
print(st.norm.interval(confidence_level,
                 loc=np.mean(fb_data),
                 scale=st.sem(fb_data)))
