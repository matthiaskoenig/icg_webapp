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

## Documentation
Inputs:
- age
- body weight
- CPT score/preoperative ICG-R15
- liver volume (NA)
- liver blood flow (NA)/cardiac output

```
age = 55  # [yr] (min 18, max 84)
bodyweight = 75  # [kg] (min 30, max 140)
ctp = "healthy"  # [healthy, A, B, C]
f_cirrhosis =  [0, ]

cirrhosis_map = {
    "Healthy": 0,
    "CTP A (mild cirrhosis)": 0.4086734693877552,
    "CTP B (moderate cirrhosis)": 0.7025510204081633,
    "CTP C (severe cirrhosis)": 0.8173469387755102
}
```
