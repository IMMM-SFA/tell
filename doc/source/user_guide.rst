==========
User Guide
==========


Setup
-----
The following with introduce you to the input data required by **tell** and how to set up a configuration file to run **tell**.


Tutorial
---------
Jupyter Notebooks


Quickstarter
~~~~~~~~~~~~
The following is a link to a Jupyter Notebook to run **tell**:  `quickstarter <https://github.com/IMMM-SFA/cerf/blob/main/notebooks/quickstarter.ipynb>`_


Fundamental Concepts
--------------------
The following are the building blocks of how **tell** projects future loads.


How It Works
~~~~~~~~~~~~
The basic logic for **tell** proceeds in six sequential steps. Note that you may not need to repeat each step (e.g., training the empirical models) each time you
want to conduct a simulation using **tell**.

#. Formulate empirical models that relate the historical observed meteorology and population to the hourly time-series of total electricity demand for each of the balancing authorities (BA) that report their hourly loads in the EIA-930 dataset.

#. Use the empirical models to predict future hourly loads for each BA based on IM3’s climate and population forcing scenarios.

#. Distribute the hourly loads for each BA to the counties that BA operates in and then aggregate the county-level hourly loads from all BAs into annual state-level loads.

#. Calculate state-level scaling factors that force the bottom-up annual state-level total loads from **tell** to match the future annual state-level total loads from GCAM-USA.

#. Apply the state scaling factors to each county-level time-series of hourly total electricity loads.

#. Output yearly time-series of total electricity demand at the state, county, and BA level that are conceptually and quantitatively consistent with each other.


Design Constraints
~~~~~~~~~~~~~~~~~~
The **tell** model was designed using the following conceptual constraints:

.. list-table::
    :header-rows: 1

    * - Topic
      - Requirement
    * - Spatial resolution and scope
      - Should cover the entire U.S. (excluding Alaska and Hawaii) and produce demands at an appropriately high spatial resolution for input into a nodal unit commitment/economic dispath (UC/ED) model
    * - Temporal resolution and scope
      - Should produce hourly projections of total electricity demand in one-year incremenets through the year 2100.
    * - Forcing factors
      - Projections should respond to changes in meteorology/climate and population.
    * - Multiscale consistency
      - Should produce hourly total electricity demand at the county, state, and balancing authority scale that are conceptually and quantitatively consistent.
    * - Open-source
      - Should be based entirely on publicly available data and be made available as an open-source model.


Balancing Authorities
~~~~~~~~~~~~~~~~~~~~~
The core predictions of **tell** occur at the scale of Balancing Authorities (BAs). BAs are responsible for the real-time balancing of electricity supply and demand within a given region of the electric grid.
For **tell**, BAs are useful because they represent the finest scale for which historical hourly load data is uniformly available across the U.S. This allows us to build an electric load forecasting
model that works across the entire country. **tell** uses historical (2015-2020) hourly load data from the `EIA-930 <https://www.eia.gov/electricity/gridmonitor/about>`_ dataset for BAs across the U.S. We note
that some smaller BAs are not included in the EIA-930 dataset. Other BAs are generation only or we were unable to geolocate them. Eight BAs (CISO, ERCO, MISO, ISNE, NYIS, PJM, PNM, and SWPP) started
reporting subregional loads in the EIA-930 dataset in 2018. Because we were unable to uniformly and objectively geolocate each of these subregions we opted to use the aggregate total loads for those BAs.
In total, we formulated a multi-layer perceptron (MLP) model for 55 out of the 68 BAs in the EIA-930 dataset.

.. list-table::
    :header-rows: 1

    * - BA Code
      - BA Name
      - EIA BA Number
      - Characteristics
    * - AEC
      - PowerSouth Energy Cooperative
      - 189
      - `AEC <_static/BA_Quick_Look_Plots/AEC_Quick_Look_Plots.png>`_
    * - AECI
      - Associated Electric Cooperative Incorporated
      - 924
      - `AECI <_static/BA_Quick_Look_Plots/AECI_Quick_Look_Plots.png>`_
    * - AVA
      - Avista Corporation
      - 20169
      - `AVA <_static/BA_Quick_Look_Plots/AVA_Quick_Look_Plots.png>`_
    * - **AVRN**
      - **Avangrid Renewables**
      - **NA**
      - **Generation Only**
    * - AZPS
      - Arizona Public Service Company
      - 803
      - `AZPS <_static/BA_Quick_Look_Plots/AZPS_Quick_Look_Plots.png>`_
    * - BANC
      - Balancing Authority of Northern California
      - 16534
      - `BANC <_static/BA_Quick_Look_Plots/BANC_Quick_Look_Plots.png>`_
    * - BPAT
      - Bonneville Power Administration
      - 1738
      - `BPAT <_static/BA_Quick_Look_Plots/BPAT_Quick_Look_Plots.png>`_
    * - CHPD
      - Public Utility District No. 1 of Chelan County
      - 3413
      - `CHPD <_static/BA_Quick_Look_Plots/CHPD_Quick_Look_Plots.png>`_
    * - CISO
      - California Independent System Operator
      - 2775
      - `CISO <_static/BA_Quick_Look_Plots/CISO_Quick_Look_Plots.png>`_
    * - CPLE
      - Duke Energy Progress East
      - 3046
      - `CPLE <_static/BA_Quick_Look_Plots/CPLE_Quick_Look_Plots.png>`_
    * - **CPLW**
      - **Duke Energy Progress West**
      - **NA**
      - **Not Geolocated**
    * - **DEAA**
      - **Arlington Valley**
      - **NA**
      - **Generation Only**
    * - DOPD
      - Public Utility District No. 1 of Douglas County
      - 5326
      - `DOPD <_static/BA_Quick_Look_Plots/DOPD_Quick_Look_Plots.png>`_
    * - DUK
      - Duke Energy Carolinas
      - 5416
      - `DUK <_static/BA_Quick_Look_Plots/DUK_Quick_Look_Plots.png>`_
    * - **EEI**
      - **Electric Energy Incorporated**
      - **NA**
      - **Generation Only**
    * - EPE
      - El Paso Electric Company
      - 5701
      - `EPE <_static/BA_Quick_Look_Plots/EPE_Quick_Look_Plots.png>`_
    * - ERCO
      - Electric Reliability Council of Texas
      - 5723
      - `ERCO <_static/BA_Quick_Look_Plots/ERCO_Quick_Look_Plots.png>`_
    * - FMPP
      - Florida Municipal Power Pool
      - 14610
      - `FMPP <_static/BA_Quick_Look_Plots/FMPP_Quick_Look_Plots.png>`_
    * - FPC
      - Duke Energy Florida
      - 6455
      - `FPC <_static/BA_Quick_Look_Plots/FPC_Quick_Look_Plots.png>`_
    * - FPL
      - Florida Power and Light
      - 6452
      - `FPL <_static/BA_Quick_Look_Plots/FPL_Quick_Look_Plots.png>`_
    * - GCPD
      - Public Utility District No. 2 of Grant County
      - 14624
      - `GCPD <_static/BA_Quick_Look_Plots/GCPD_Quick_Look_Plots.png>`_
    * - **GLHB**
      - **GridLiance**
      - **NA**
      - **Not Geolocated**
    * - **GRID**
      - **Gridforce Energy Management**
      - **NA**
      - **Generation Only**
    * - **GRIF**
      - **Griffith Energy**
      - **NA**
      - **Generation Only**
    * - **GRMA**
      - **Gila River Power**
      - **NA**
      - **Generation Only**
    * - GVL
      - Gainesville Regional Utilities
      - 6909
      - `GVL <_static/BA_Quick_Look_Plots/GVL_Quick_Look_Plots.png>`_
    * - **GWA**
      - **NaturEner Power Watch**
      - **NA**
      - **Generation Only**
    * - **HGMA**
      - **New Harquahala Generating Company**
      - **NA**
      - **Generation Only**
    * - HST
      - City of Homestead
      - 8795
      - `HST <_static/BA_Quick_Look_Plots/HST_Quick_Look_Plots.png>`_
    * - IID
      - Imperial Irrigation District
      - 9216
      - `IID <_static/BA_Quick_Look_Plots/IID_Quick_Look_Plots.png>`_
    * - IPCO
      - Idaho Power Company
      - 9191
      - `IPCO <_static/BA_Quick_Look_Plots/IPCO_Quick_Look_Plots.png>`_
    * - ISNE
      - Independent System Operator of New England
      - 13434
      - `ISNE <_static/BA_Quick_Look_Plots/ISNE_Quick_Look_Plots.png>`_
    * - JEA
      - JEA
      - 9617
      - `JEA <_static/BA_Quick_Look_Plots/JEA_Quick_Look_Plots.png>`_
    * - LDWP
      - Los Angeles Department of Water and Power
      - 11208
      - `LDWP <_static/BA_Quick_Look_Plots/LDWP_Quick_Look_Plots.png>`_
    * - LGEE
      - Louisville Gas and Electric Company and Kentucky Utilities Company
      - 11249
      - `LGEE <_static/BA_Quick_Look_Plots/LGEE_Quick_Look_Plots.png>`_
    * - MISO
      - Midcontinent Independent System Operator
      - 56669
      - `MISO <_static/BA_Quick_Look_Plots/MISO_Quick_Look_Plots.png>`_
    * - NEVP
      - Nevada Power Company
      - 13407
      - `NEVP <_static/BA_Quick_Look_Plots/NEVP_Quick_Look_Plots.png>`_
    * - NSB
      - Utilities Commission of New Smyrna Beach
      - 13485
      - `NSB <_static/BA_Quick_Look_Plots/NSB_Quick_Look_Plots.png>`_
    * - NWMT
      - NorthWestern Corporation
      - 12825
      - `NWMT <_static/BA_Quick_Look_Plots/NWMT_Quick_Look_Plots.png>`_
    * - NYIS
      - New York Independent System Operator
      - 13501
      - `NYIS <_static/BA_Quick_Look_Plots/NYIS_Quick_Look_Plots.png>`_
    * - **OVEC**
      - **Ohio Valley Electric Corporation**
      - **NA**
      - **Retired**
    * - PACE
      - PacifiCorp East
      - 14379
      - `PACE <_static/BA_Quick_Look_Plots/PACE_Quick_Look_Plots.png>`_
    * - PACW
      - PacifiCorp West
      - 14378
      - `PACW <_static/BA_Quick_Look_Plots/PACW_Quick_Look_Plots.png>`_
    * - PGE
      - Portland General Electric Company
      - 15248
      - `PGE <_static/BA_Quick_Look_Plots/PGE_Quick_Look_Plots.png>`_
    * - PJM
      - PJM Interconnection
      - 14725
      - `PJM <_static/BA_Quick_Look_Plots/PJM_Quick_Look_Plots.png>`_
    * - PNM
      - Public Service Company of New Mexico
      - 15473
      - `PNM <_static/BA_Quick_Look_Plots/PNM_Quick_Look_Plots.png>`_
    * - PSCO
      - Public Service Company of Colorado
      - 15466
      - `PSCO <_static/BA_Quick_Look_Plots/PSCO_Quick_Look_Plots.png>`_
    * - PSEI
      - Puget Sound Energy
      - 15500
      - `PSEI <_static/BA_Quick_Look_Plots/PSEI_Quick_Look_Plots.png>`_
    * - SC
      - South Carolina Public Service Authority
      - 17543
      - `SC <_static/BA_Quick_Look_Plots/SC_Quick_Look_Plots.png>`_
    * - SCEG
      - South Carolina Electric and Gas Company
      - 17539
      - `SCEG <_static/BA_Quick_Look_Plots/SCEG_Quick_Look_Plots.png>`_
    * - SCL
      - Seattle City Light
      - 16868
      - `SCL <_static/BA_Quick_Look_Plots/SCL_Quick_Look_Plots.png>`_
    * - SEC
      - Seminole Electric Cooperative
      - 21554
      - `SEC <_static/BA_Quick_Look_Plots/SEC_Quick_Look_Plots.png>`_
    * - SEPA
      - Southeastern Power Administration
      - **NA**
      - **Generation Only**
    * - SOCO
      - Southern Company Services - Transmission
      - 18195
      - `SOCO <_static/BA_Quick_Look_Plots/SOCO_Quick_Look_Plots.png>`_
    * - SPA
      - Southwestern Power Administration
      - 17716
      - `SPA <_static/BA_Quick_Look_Plots/SPA_Quick_Look_Plots.png>`_
    * - SRP
      - Salt River Project
      - 16572
      - `SRP <_static/BA_Quick_Look_Plots/SRP_Quick_Look_Plots.png>`_
    * - SWPP
      - Southwest Power Pool
      - 59504
      - `SWPP <_static/BA_Quick_Look_Plots/SWPP_Quick_Look_Plots.png>`_
    * - TAL
      - City of Tallahassee
      - 18445
      - `TAL <_static/BA_Quick_Look_Plots/TAL_Quick_Look_Plots.png>`_
    * - TEC
      - Tampa Electric Company
      - 18454
      - `TEC <_static/BA_Quick_Look_Plots/TEC_Quick_Look_Plots.png>`_
    * - TEPC
      - Tucson Electric Power
      - 24211
      - `TEPC <_static/BA_Quick_Look_Plots/TEPC_Quick_Look_Plots.png>`_
    * - TIDC
      - Turlock Irrigation District
      - 19281
      - `TIDC <_static/BA_Quick_Look_Plots/TIDC_Quick_Look_Plots.png>`_
    * - TPWR
      - City of Tacoma Department of Public Utilities
      - 18429
      - `TPWR <_static/BA_Quick_Look_Plots/TPWR_Quick_Look_Plots.png>`_
    * - TVA
      - Tennessee Valley Authority
      - 18642
      - `TVA <_static/BA_Quick_Look_Plots/TVA_Quick_Look_Plots.png>`_
    * - WACM
      - Western Area Power Administration - Rocky Mountain Region
      - 28503
      - `WACM <_static/BA_Quick_Look_Plots/WACM_Quick_Look_Plots.png>`_
    * - WALC
      - Western Area Power Administration - Desert Southwest Region
      - 25471
      - `WALC <_static/BA_Quick_Look_Plots/WALC_Quick_Look_Plots.png>`_
    * - WAUW
      - Western Area Power Administration - Upper Great Plains West
      - 19610
      - `WAUW <_static/BA_Quick_Look_Plots/WAUW_Quick_Look_Plots.png>`_
    * - **WWA**
      - **NaturEner Wind Watch**
      - **NA**
      - **Generation Only**
    * - **YAD**
      - **Alcoa Power Generating - Yadkin Division**
      - **NA**
      - **Generation Only**


Geolocating Balancing Authorities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As a spatially-explicit model, **tell** needs the ability to geolocate the loads it predicts. Since the fundamental predictions
in **tell** occur at the spatial scale of BAs, we needed to devise a way to determine where each BA operated within the U.S.
For **tell**, being able to do  this geolocation using county boundaries has a number of benefits in terms of load disaggregation
and reaggregation - so we focused on techniques to map BAs to the counties they operate in. While there are multiple maps
of BA service territories available online, there are several fundamental challenges to using maps generated by others:

1. The provenance of the data and methodology underpinning most of the maps is unknown. In other words, there is no way to determine
how the BAs were placed and if the methods used to do so are robust.

2. The maps often depict the BAs as spatially unique and non-overlapping. For county-scale mapping at least, we know this to be
untrue. Additionally, the maps are typically static representations of how BAs were configured at a single point in time. As the
actual territory of BAs can and does change over time, this presents challenges for placing BA loads occurring over a range of years.

3. Maps available online are often cartoon or stylized versions of reality with curvy lines that do not follow traditional geopolitical
boundaries. As such, to go from the cartoon map to an actual list of counties that a BA operates in would necessitate a number of
subjective decisions that are not reproducible from person to person.

Given these limitations, we devised our own method to geolocate BAs in **tell**. We eventually found a publicly available
dataset from the EIA that served as our foundation. The `EIA-861 <https://www.eia.gov/electricity/data/eia861/>`_ dataset is an annual
report describing the characteristics of the electric power industry in the U.S. Among other information, EIA-861 contains two sets of
data that are critical to **tell**:

* The "Service_Territory_YYYY.xlsx" spreadsheet provides a list of every county that a given utility operates in:

.. image:: _static/utility_to_county.png
   :width: 600
   :align: center

* The "Sales_Ult_Cust_YYYY.xlsx" spreadsheet provides the BA that each utility reports to in a given state:

.. image:: _static/utility_to_ba.png
   :width: 600
   :align: center

Using these two datasets in combination, **tell** reverse engineers the counties that each BA likely operated in within a given year. In
addition to being completely objective and reproducible, this method overcomes the limitations described above because it allows
more than one BA to be mapped to a single county and also allows the geolocation of BAs to evolve over time. **tell**
maps BA service territory annually from 2015-2019. The results of that mapping are contained in the .csv files below and are
summarized graphically in the map. The spatial extent of each BA in 2019 is shown in the link for each BA in the table above.

.. image:: _static/Overlapping_Balancing_Authorities_Map.png
   :width: 900
   :align: center

This figure shows the number of BAs that **tell** identifies as operating within each county in 2019. The bottom panel shows an example
of four different BAs reported operating in Brevard County, FL. While the majority of counties only have one BA identified, some counties
have as many as five. Note that a handful of counties had zero BAs identified as operating within them in 2019.

.. list-table::
    :header-rows: 1

    * - Year
      - Mapping File
    * - 2015
      - `Mapping <user_guide_data/fips_service_match_2015.csv>`_
    * - 2016
      - `Mapping <user_guide_data/fips_service_match_2016.csv>`_
    * - 2017
      - `Mapping <user_guide_data/fips_service_match_2017.csv>`_
    * - 2018
      - `Mapping <user_guide_data/fips_service_match_2018.csv>`_
    * - 2019
      - `Mapping <user_guide_data/fips_service_match_2019.csv>`_


Load Disaggregation and Reaggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**tell** uses multiple instances of load disaggregation and reaggregation in order to cross spatiotemporal scales. The fundamental
predictions in **tell** occur at the spatial scale of BAs. In order to compare those hourly load values at the BA-level with the
annual state-level load values produced by GCAM-USA we first disaggregate the hourly predicted BA-level loads to the county-level
and then reaggregate those hourly county-level loads to an annual total load prediction for each state. For each BA we identify
the counties that BA operates in using the methodology described above. We then use the
county-level populations for those counties to determine the fraction of the BA's total load that should be assigned to each county. A
graphical depiction of this for the ISNE BA is shown below. Using this approach, the load received by each county in a BA's service territory has the
same shape and temporal patterns, but the magnitude varies depending on the population in that county relative to the total population
in the BA's service territory. As there are spatial overlaps in BAs, many counties receive partial loads from more than one BA.

.. image:: _static/Load_Projection_Dissagregation_Example_ISNE.png
   :width: 900
   :align: center

Once the load projections from all BAs in **tell** have been disaggregated to the county-level, we next sum up the loads from all
counties in a given state to get annual total state-level loads which are scaled to match the projections from GCAM-USA. The scaling
factors for each state are then applied to all county-level hourly load values in that state. The final output of **tell** is thus
a series of 8760-hr time series of total electricity loads at the state-, county-, and BA-level that are conceptually and quantitatively
consistent with one another.

It is important to note that the future evolution of population is also taken into account in **tell**. Projected annual changes in
population for each county and state are generated using the SSP scenarios. Those future populations are used in post-processing the
MLP models and to derive new weighting factors to be used in disaggregating and reaggregating future **tell** loads.
Thus, in an example scenario where lots of people move to Southern California, the counties there would not only receive a higher
proportion of the BA-level loads for BAs operating there, but would also have an incrementally larger impact on the future total
hourly load profile for California as a whole.


Multi-Layer Perceptron (MLP) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**tell** uses a series of multilayer perceptron (MLP) models to predict future loads. There is one unique MLP model for each BA. The
general form of each MLP model is:

.. math::

   y_{pred} = y_{MLP} + `\epsilon`

where y :subscript:`MLP` is the actual MLP model and epsilon represents a linear model that uses the annual evolution of total population
within the BA service territory to predict the residuals from the actual MLP model for a given BA. The MLP model for each BA is trained and
evaluated independently. Hyperparameter tuning for the models is done using grid search. The MLP models are trained on historical load data
from the `EIA-930 <https://www.eia.gov/electricity/gridmonitor/about>`_ dataset and weather from IM3's historical runs using the Weather
Research and Forecasting (WRF) model. In the production version of the **tell** model the MLP models for each BA were trained on data from
2016-2018 and evaluated against observed loads from 2019. For the **tell** quickstarter notebook the MLP models are trained and evaluated against data
from 2019 only in order to improve the timeliness of the training process. Details of the MLP predictive variables are included in the table below.

.. list-table::
    :header-rows: 1

    * - Variable
      - Description
      - Units/Format
    * - Temperature
      - 2-m temperature from WRF (T2)
      - K
    * - Specific humidity
      - 2-m water vapor mixing ratio from WRF (Q2)
      - kg kg :sup:`-1`
    * - Shortwave radiation
      - Downwelling shortwave radiative flux at the surface from WRF (SWdn)
      - W m :sup:`-2`
    * - Longwave radiation
      - Downwwelling longwave radiative flux at the surface from WRF (GLW)
      - W m :sup:`-2`
    * - Wind speed
      - 10-m wind speed derived from the U and V wind components from WRF (U10 and V10)
      - m s :sup:`-1`
    * - Population
      - Total population in the counties covered by the BA
      - NA
    * - Day of the week
      - Day of the week
      - Weekday or weekend
    * - Hour of the day
      - Hour of the day in UTC
      - 00-23
    * - Federal holiday
      - Is the day a federal holiday?
      - Yes/No


Scenarios
~~~~~~~~~
**tell** is designed to work in conjunction with the United States version of the Global Change Analysis Model (GCAM-USA)
to explore different future scenarios of population and climate change. The models are configured to run the following
combinations of Representative Concentration Pathways (`RCPs <https://en.wikipedia.org/wiki/Representative_Concentration_Pathway>`_)
and Shared Socioeconomic Pathways (`SSPs <https://en.wikipedia.org/wiki/Shared_Socioeconomic_Pathways>`_):

.. list-table::
    :header-rows: 1

    * - Climate Scenario
      - Population Scenario
      - scenario_name
    * - RCP 4.5 - Cooler
      - SSP3
      - rcp45cooler_ssp3
    * - RCP 4.5 - Cooler
      - SSP5
      - rcp45cooler_ssp5
    * - RCP 4.5 - Hotter
      - SSP3
      - rcp45hotter_ssp3
    * - RCP 4.5 - Hotter
      - SSP5
      - rcp45hotter_ssp5
    * - RCP 8.5 - Cooler
      - SSP3
      - rcp85cooler_ssp3
    * - RCP 8.5 - Cooler
      - SSP5
      - rcp85cooler_ssp5
    * - RCP 8.5 - Hotter
      - SSP3
      - rcp85hotter_ssp3
    * - RCP 8.5 - Hotter
      - SSP5
      - rcp85hotter_ssp5


Key Outputs
-----------
**tell** produces four types of output files. Each type of output is written out as a .csv file or series of .csv files in the ``output_directory``.
Each type of output file can be suppressed by commenting out the relevant line in ``execute_forward.py``. Missing values in each output file are
coded as -9999. All times are in UTC.


State Summary Data
~~~~~~~~~~~~~~~~~~
This output file gives the annual total loads for each of the 48 states in the CONUS and the District of Columbia. It also contains the scaling factor for
each state that force the aggregate annual total loads from  **tell** to agree with those produced by GCAM-USA.

Filename: *TELL_State_Summary_Data_YYYY.csv*

.. list-table::
    :header-rows: 1

    * - Name
      - Description
      - Units/Format
    * - Year
      - Year of load
      - NA
    * - State_Name
      - Name of the state
      - NA
    * - State_FIPS
      - `FIPS <https://www.census.gov/library/reference/code-lists/ansi.html>`_ code of the state
      - NA
    * - State_Scaling_Factor
      - Scaling factor to force agreement between **tell** and GCAM-USA annual total loads
      - NA
    * - GCAM_USA_Load_TWh
      - Annual total load for the state from GCAM-USA
      - TWh
    * - Raw_TELL_Load_TWh
      - Unscaled annual total load for the state from TELL
      - TWh
    * - Scaled_TELL_Load_TWh
      - Scaled annual total load for the state from TELL
      - TWh


State Hourly Load Data
~~~~~~~~~~~~~~~~~~~~~~
This output file gives the hourly time-series of total loads for each of the 48 states in the CONUS and the District of Columbia.

Filename: *TELL_State_Hourly_Load_Data_YYYY.csv*

.. list-table::
    :header-rows: 1

    * - Name
      - Description
      - Units/Format
    * - State_Name
      - Name of the state
      - NA
    * - State_FIPS
      - `FIPS <https://www.census.gov/library/reference/code-lists/ansi.html>`_ code of the state
      - NA
    * - Time_UTC
      - Hour of the load in UTC
      - YYYY-MM-DD HH:MM:SS
    * - Raw_TELL_State_Load_MWh
      - Unscaled hourly total load for the state from TELL
      - MWh
    * - Scaled_TELL_State_Load_MWh
      - Scaled hourly total load for the state from TELL
      - MWh


Balancing Authority Hourly Load Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This output file gives the hourly time-series of total loads for each of the BAs simulated by **tell**.

Filename: *TELL_Balancing_Authority_Hourly_Load_Data_YYYY.csv*

.. list-table::
    :header-rows: 1

    * - Name
      - Description
      - Units/Format
    * - BA_Code
      - Alphanumeric code for the BA
      - NA
    * - BA_Number
      - Designated EIA number for the BA
      - NA
    * - Time_UTC
      - Hour of the load in UTC
      - YYYY-MM-DD HH:MM:SS
    * - Raw_TELL_BA_Load_MWh
      - Unscaled hourly total load for the BA from TELL
      - MWh
    * - Scaled_TELL_BA_Load_MWh
      - Scaled hourly total load for the BA from TELL
      - MWh


County Hourly Load Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This collection of output files gives the hourly time-series of total loads for each county in the CONUS and the District of Columbia.
These output files are stored in a subdirectory of ``output_directory`` named ``County_Level_Data``.

Filename Format: *TELL_statename_countyname_Hourly_Load_Data_YYYY.csv*

.. list-table::
    :header-rows: 1

    * - Name
      - Description
      - Units/Format
    * - County_Name
      - Name of the county
      - NA
    * - County_FIPS
      - `FIPS <https://www.census.gov/library/reference/code-lists/ansi.html>`_ code of the county
      - NA
    * - State_Name
      - Name of the state the county is in
      - NA
    * - State_FIPS
      - `FIPS <https://www.census.gov/library/reference/code-lists/ansi.html>`_ code of the state
      - NA
    * - Time_UTC
      - Hour of the load in UTC
      - YYYY-MM-DD HH:MM:SS
    * - Raw_TELL_County_Load_MWh
      - Unscaled hourly total load for the county from TELL
      - MWh
    * - Scaled_TELL_County_Load_MWh
      - Scaled hourly total load for the county from TELL
      - MWh