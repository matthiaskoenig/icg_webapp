# Inputs:
# - age
# - body weight
# - CPT score/preoperative ICG-R15 -> f_cirrhosis (response curve; response curve mappings)
# - liver volume (NA)
# - liver blood flow (NA)/cardiac output

age = 55  # [yr] (min 18, max 84)
bodyweight = 75  # [kg] (min 30, max 140)
# FIXME: alternative preoperative ICG-R15
ctp = "healthy"  # [NA, healthy, A, B, C]

# FIXME: can be NA (handle this case in the sampling)
liver_volume = 1.5  # [l] (min 0.2, max 3.0)
# FIXME: can be NA (handle this case in the sampling)
# FIXME: alternative cardiac output
hepatic_bloodflow = 1.0  # [l/min] (min 0.2, max 3.0)

# Sampling:
# Sample 100 representation of the individual
# - liver volume (sample if NA, use 1D/2D sampling)
# - liver blood flow  (sample if NA, use 1D/2D sampling)
# - oatp1b3 (sample)






# here we have 100 samples;

# Calculate post-operative ICG-R15 results

# run simulation with various resection rate [0, 10, ... ,90]
# - for every resection rate we get a distribution of postoperative ICG-R15
# - display information which supports decision
# => boxplot of ICG-R15, ICG-PDR

# translate in survival
# - => post-operative ICG-R15 can be tranlated into probability of survival via 1D classification model
# - plot survival probability vs resection