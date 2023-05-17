# Indocyanine Green ICG liver function webapp
Streamlit web application for the indocyanine green model.

## Setup environment
To run the example applications install the requirements 
```
cd icg_webapp
mkvirtualenv icg_webapp --python=python3.9
(icg_webapp) pip install -r requirements.txt --upgrade
```

## Run application
To run the app use:
```
streamlit run icg_app.py
``` 



## Misc documentation
Inputs:
- age
- body weight
- CPT score/preoperative ICG-R15 -> f_cirrhosis (response curve; response curve mappings)
- liver volume (NA)
- liver blood flow (NA)/cardiac output

age = 55  # [yr] (min 18, max 84)
bodyweight = 75  # [kg] (min 30, max 140)

# FIXME: alternative preoperative ICG-R15
ctp = "healthy"  # [healthy, A, B, C]
f_cirrhosis =  [0, ]

cirrhosis_map = {
    "Healthy": 0,
    "CTP A (mild cirrhosis)": 0.4086734693877552,
    "CTP B (moderate cirrhosis)": 0.7025510204081633,
    "CTP C (severe cirrhosis)": 0.8173469387755102
}
#
# # FIXME: can be NA (handle this case in the sampling)
# liver_volume = 1.5  # [l] (min 0.2, max 3.0)
# # FIXME: can be NA (handle this case in the sampling)
# # FIXME: alternative cardiac output
# hepatic_bloodflow = 1.0  # [l/min] (min 0.2, max 3.0)

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

# for resection_rate in [0, 0.5]:
resection_rate = 0.5

# run simulation & calculate ICG-R15




# translate in survival
# - => post-operative ICG-R15 can be tranlated into probability of survival via 1D classification model
# - plot survival probability vs resection
