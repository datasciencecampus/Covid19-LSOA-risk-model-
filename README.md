# COVID-19 LSOA Risk Model 

The COVID-19 LSOA risk model project aims to infer the risk factors associated with COVID-19 incidence at the community (LSOA) level. Weekly COVID-19 incidence is modelled from April 2020 to the latest date for which data on the number of confirmed cases is available from NHS Test & Trace.

The model reports coefficient estimates for the features associated with COVID-19 incidence.

## Getting Started

Clone this repository and install packages from the `requirements.txt` file

## Running the model

* Confirm that you have access to the required data sets, whose location is presented in Section A of the config file which can be found in `src` > `utils` > `config.py`

* Run the scripts from the `src` > `pipeline` folder in the following order:

1. 01_Preprocess.py
2. 02_Model.py
3. 03_Output.py

At the end of the `02_Model.py` script, the modelling results are written to the Google BigQuery locations specified in Section B of the config file.

The script `03_Output.py` wrangles the model results into a format for presentation on a [Google Data Studio dashboard](https://datastudio.google.com/reporting/8a6f31bd-1510-48dc-b314-f52419f75a3b/page/p_zxqqfjf9pc) which can be accessed by anyone with an @ons.gov.uk email address. If updating the dashboard is not a requirement for your use case, then it is not necessary to execute the `03_Output.py` script.  

### Points of contact at the Data Science Campus for this project

Kaveh Jahanshahi: Kaveh.Jahanshahi@ons.gov.uk  
Chaitanya Joshi: Chaitanya.Joshi@ons.gov.uk

### Acknowledgements

The refactored codebase was developed by Daniel Morgan and Lilith Barca from Methods Analytics with permission to publish any code alongside the academic publication for this project. For more information please contact lilith.barca@methods.co.uk

