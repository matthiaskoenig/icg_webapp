from typing import Dict

import streamlit as st
import xarray
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt

from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial as Simulator

from pathlib import Path
icg_model_path = Path(__file__).parent / "model" / "icg_body_flat.xml"


def load_model(model_path: Path) -> Simulator:
    """Load model."""
    return Simulator(model_path)


simulator = load_model(icg_model_path)


def simulate(start: float, end: float, steps: int, changes: Dict) -> xarray.Dataset:
    """Simulate model."""
    tc_sim = TimecourseSim(
        [
            Timecourse(start=start, end=end, steps=int(steps),
                       changes=changes)
         ]
    )
    return simulator.run_timecourse(tc_sim)


st.sidebar.title('Settings')
st.sidebar.markdown("## Integration")
endtime = st.sidebar.number_input("End time [min]", value=20, min_value=1,
                                  max_value=200, step=10)
steps = st.sidebar.number_input("Steps", value=100, min_value=2, max_value=300,
                                step=10)

st.sidebar.markdown("## Parameters")
icg_dose = st.sidebar.slider("ICG Dose [mg]", value=10, min_value=0, max_value=200,
                             step=5)

'''
# Physiological based pharmacokinetics model of indocyanine green
Personalized prediction of survival after hepatectomy.
'''

# st.image(
#     image="./model/icg_body.png",
#     width=600,
#     caption="Model overview: A: PBPK model. The whole-body model for ICG consists "
#             "of venous blood, arterial blood, lung, liver, gastrointestinal tract, "
#             "and rest compartment (accounting for organs not modeled in detail). "
#             "The systemic blood circulation connects these compartments. "
#             "B: Liver model. ICG is taken up into the liver tissue (hepatocytes) "
#             "via OATP1B3. The transport was modeled as competitively inhibited by "
#             "plasma bilirubin. Hepatic ICG is excreted in the bile from where it is "
#             "subsequently excreted in the feces. No metabolism of ICG occurs in "
#             "the liver.")

'''
## Results
'''
xr = simulate(start=0, end=endtime, steps=steps, changes={"IVDOSE_icg": icg_dose})
# print(xr)

# select simulation results
selections = ["time", "[Cve_icg]", "[Car_icg]"]
data = [xr[key].values[:, 0] for key in selections]
df = pd.DataFrame(dict(zip(selections, data)))
# print(df.head(10))




fig, (ax1, ax2) = plt.subplots(figsize=(7, 5), nrows=1, ncols=2)
ax1.plot(df.time, df["[Cve_icg]"], '-o', color="blue", alpha=0.7)
ax1.plot(df.time, df["[Car_icg]"], '-o', color="black", alpha=0.7)
ax1.set_xlabel("time [min]")
ax1.set_ylabel("concentration [mmole/min]")
ax2.plot(df["[Car_icg]"], df["[Cve_icg]"], '-o', color="blue", alpha=0.7)

ax2.set_xlim(0, 0.10)
for ax in [ax1, ax2]:
    ax.set_ylim(0, 0.10)
    ax.grid(True)

st.pyplot(fig=fig, clear_figure=False)

# raw data
if st.checkbox('Show simulation data'):
    st.subheader('Raw data')
    st.write(df)

'''
## References
**Physiologically based modeling of the effect of physiological and anthropometric variability on indocyanine green based liver function tests**  
*Adrian Köller, Jan Grzegorzewski and Matthias König*  
bioRxiv 2021.08.11.455999; doi: https://doi.org/10.1101/2021.08.11.455999  
[Submitted to Frontiers Physiology 2021-08-12]

**Prediction of survival after hepatectomy using a physiologically based pharmacokinetic model of indocyanine green liver function tests**  
*Adrian Köller, Jan Grzegorzewski, Michael Tautenhahn, Matthias König*  
bioRxiv 2021.06.15.448411; doi: https://doi.org/10.1101/2021.06.15.448411,  
[Submitted to Frontiers Physiology 2021-06-30]
'''
st.markdown("## Disclaimer")
st.caption("The software is provided **AS IS**, without warranty of any kind, express or implied, "
           "including but not limited to the warranties of merchantability, "
           "fitness for a particular purpose and noninfringement. In no event shall the "
           "authors or copyright holders be liable for any claim, damages or other liability, "
           "whether in an action of contract, tort or otherwise, arising from, out of or in "
           "connection with the software or the use or other dealings in the software. "
           ""
           "This software is a research proof-of-principle and not fit for any clinical application."
           )
