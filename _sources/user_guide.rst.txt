==========
User Guide
==========
This user guide is meant to explain how **tell** works and the concepts that it is built upon. More information about how to
run the model can be found in the **tell** `quickstarter <https://github.com/IMMM-SFA/tell/blob/main/notebooks/tell_quickstarter.ipynb>`_
notebook that contains detailed step-by-step instructions on how to run **tell**.


About **tell**
--------------
The Total ELectricity Load (**tell**) model projects the short- and long-term evolution of hourly electricity demand in response to future changes in
weather and climate. The purpose of **tell** is to generate end-of-century hourly profiles of electricity demand across the entire Conterminous
United States (CONUS) at a spatial resolution adequate for input to a unit commitment/economic dispatch (UC/ED) model while also maintaining consistency
with the long-term growth and evolution of annual state-level electricity demand projected by an economically driven human-Earth system model. **tell** takes
as input future projections of the hourly time-series of meteorology and decadal populations and uses the temporal variations in weather to project
hourly profiles of total electricity demand. The core predictions in **tell** are based on a series of multilayer perceptron (MLP) models for 54 independent
Balancing Authorities (BAs). Those MLP models are trained on historical observations of weather and electricity demand. Hourly projections from **tell**
are scaled to match the annual state-level total electricity loads projected by the U.S. version of the Global Change Analysis Model (GCAM-USA) which
captures the long-term co-evolution of the human-Earth system. Using this unique approach allows **tell** to reflect both changes in the shape
of the load profile due to variations in weather and the long-term evolution of energy demand due to changes in population, technology, and economics.
**tell** is unique from other load forecasting models in that it features an explicit spatial component that allows it to relate projected
loads to where they would occur spatially within a grid operations model. The output of **tell** is a series of hourly projections of future electricity
demand at the county-, state-, and BA-scale that are conceptually and quantitatively consistent with one another.

**tell** was designed to work using data from 54 BAs the U.S. and in conjunction with the GCAM-USA model. Thus, it is
not immediately extensible to other countries (e.g., in Europe). However, the fundamental modeling approach based on MLP
models trained on historical loads and meteorology data could easily be adapted to work in other regions with sufficient
data.


How It Works
------------
The basic workflow for **tell** proceeds in six sequential steps:

#. Formulate empirical models that relate the historical observed meteorology to the hourly time-series of total electricity demand for 54 BAs that report their hourly loads in the `EIA-930 <https://www.eia.gov/electricity/gridmonitor/about>`_ dataset.

#. Use the empirical models to project future hourly loads for each BA based on IM3’s climate and population scenarios.

#. Distribute the hourly loads for each BA to the counties that BA operates in and then aggregate the county-level hourly loads from all BAs into annual state-level loads.

#. Calculate annual state-level scaling factors that force the bottom-up annual state-level total loads from **tell** to match the annual state-level total loads from GCAM-USA.

#. Apply the state-level scaling factors to each county- and BA-level time-series of hourly total demand.

#. Output yearly 8760-hr time-series of total electricity demand at the state-, county-, and BA-scale that are conceptually and quantitatively consistent with each other.


Design Constraints
------------------
**tell** was designed with the following constraints:

.. list-table::
    :header-rows: 1

    * - Topic
      - Requirement
    * - Spatial resolution and scope
      - Should cover the entire U.S. (excluding Alaska and Hawaii) and produce demands at an appropriately high spatial resolution for input into a nodal UC/ED model
    * - Temporal resolution and scope
      - Should produce hourly projections of total electricity demand in one-year increments through the year 2100.
    * - Forcing factors
      - Projections should respond to changes in meteorology and climate.
    * - Multiscale consistency
      - Should produce hourly total electricity demand at the county-, state-, and BA-scale that are conceptually and quantitatively consistent with each other.
    * - Open-source
      - Should be based entirely on publicly available data and be made available as an extensively-documented open-source model.


Fundamental Concepts
--------------------
The following are the building blocks of how **tell** projects future loads.


Balancing Authorities
~~~~~~~~~~~~~~~~~~~~~
The core projections of **tell** occur at the scale of Balancing Authorities (BAs). BAs are responsible for the real-time balancing of electricity supply and demand within a given region of the electric grid.
For **tell**, BAs are useful because they represent the finest scale for which historical hourly load data is uniformly available across the U.S. This allows us to build an electric load projection
model that works across the entire country. **tell** uses historical (2015-2019) hourly load data from the `EIA-930 <https://www.eia.gov/electricity/gridmonitor/about>`_ dataset for BAs across the U.S. We note
that some smaller BAs are not included in the EIA-930 dataset. Other BAs are generation only or we were unable to geolocate them. Eight BAs (CISO, ERCO, MISO, ISNE, NYIS, PJM, PNM, and SWPP) started
reporting subregional loads in the EIA-930 dataset in 2018. Because we were unable to uniformly and objectively geolocate each of these subregions we opted to use the aggregate total loads for those BAs.
In total, we formulated a multi-layer perceptron (MLP) model for 54 out of the 68 BAs in the EIA-930 dataset.

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
As a spatially-explicit model, **tell** needs the ability to geolocate the loads it projects. Since the fundamental projections
in **tell** occur at the spatial scale of BAs, we needed to devise a way to determine where each BA operated within the U.S.
For **tell**, being able to do  this geolocation using county boundaries has a number of benefits in terms of load disaggregation
and reaggregation - so we focused on techniques to map BAs to the counties they operate in. While there are multiple maps
of BA service territories available online, there are several fundamental challenges to using maps generated by others:

1. The provenance of the data and methodology underpinning most of the maps is unknown. In other words, there is no way to determine
how the BAs were placed and if the methods used to do so are robust.

2. The maps often depict the BAs as spatially unique and non-overlapping. For a county-scale mapping at least, we know this to be
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
maps BA service territory annually from 2015-2020. The results of that mapping are summarized graphically in the map below.
The spatial extent of each BA in 2019 is shown in the link for each BA in the table above.

.. image:: _static/Overlapping_Balancing_Authorities_Map.png
   :width: 900
   :align: center

This figure shows the number of BAs that **tell** identifies as operating within each county in 2019. The bottom panel shows an example
of four different BAs reported operating in Brevard County, FL. While the majority of counties only have one BA identified, some counties
have as many as five. Note that a handful of counties had zero BAs identified as operating within them in 2019. Because we think these
BA-to-county mappings may be useful to many others the output files from the mapping process are included as .csv files below. They can be
reproduced within the **tell** package by running the ``tell.map_ba_service_territory`` function.

.. list-table::
    :header-rows: 1

    * - Year
      - File
    * - 2015
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2015.csv>`_
    * - 2016
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2016.csv>`_
    * - 2017
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2017.csv>`_
    * - 2018
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2018.csv>`_
    * - 2019
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2019.csv>`_
    * - 2020
      - `Mapping <_static/User_Guide_Data/ba_service_territory_2020.csv>`_


Load Disaggregation and Reaggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**tell** uses multiple instances of load disaggregation and reaggregation in order to cross spatiotemporal scales. The fundamental
projections in **tell** occur at the spatial scale of BAs. In order to compare those hourly load values at the BA-level with the
annual state-level load values produced by GCAM-USA, we first disaggregate the hourly projected BA-level loads to the county-level
and then reaggregate those hourly county-level loads to an annual total load projection for each state. For each BA we identify
the counties that BA operates in using the methodology described above. We then use the county-level populations for those counties
to determine the fraction of the BA's total load that should be assigned to each county. A graphical depiction of this for the ISNE BA
is shown below. Using this approach, the load received by each county in a BA's service territory has the same shape and temporal
patterns, but the magnitude varies depending on the population in that county relative to the total population in the BA's service
territory. As there are spatial overlaps in BAs, many counties receive partial loads from more than one BA.

.. image:: _static/Load_Projection_Dissagregation_Example_ISNE.png
   :width: 900
   :align: center

Once the load projections from all BAs in **tell** have been disaggregated to the county-level, we next sum up the loads from all
counties in a given state to get annual state-level total loads which are scaled to match the projections from GCAM-USA. The scaling
factors for each state are then applied to all county- and BA-level hourly load values in that state. The final output of **tell** is thus
a series of 8760-hr time series of total electricity loads at the county-, state-, and BA-scale that are conceptually and quantitatively
consistent with each other.

It is important to note that the future evolution of population is also taken into account in **tell**. Projected annual changes in
population for each county and state are generated using the Shared Socioeconomic Pathways (`SSPs <https://en.wikipedia.org/wiki/Shared_Socioeconomic_Pathways>`_)
scenarios. Those future populations are used to derive new weighting factors to be used in disaggregating and reaggregating future **tell** loads.
Thus, in an scenario where lots of people move to, for example, Southern California, the counties there would not only receive a higher
proportion of the BA-level loads for BAs operating there, but would also have an incrementally larger impact on the future total
hourly load profile for California as a whole.


Multilayer Perceptron (MLP) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**tell** uses a series of multilayer perceptron (MLP) models to project future loads. There is one unique MLP model for each BA. The MLP model
for each BA is trained and evaluated independently. The MLP models are trained on historical load data from the `EIA-930 <https://www.eia.gov/electricity/gridmonitor/about>`_
dataset and weather from IM3's historical runs using the Weather Research and Forecasting (WRF) model. In the production version of **tell**
the MLP models for each BA were trained on data from 2016-2018 and evaluated against observed loads from 2019. While the EIA-930 data extends past
the year 2019, COVID-19 induced significant changes in the diurnal profile of electricity demand (e.g., `Burleyson et al. 2021 <https://www.sciencedirect.com/science/article/pii/S0306261921010631>`_)
so we opted not to use 2020+ data in the MLP model training or evaluation. In the future, **tell** could be retrained repeatedly as more and
more EIA-930 data becomes available.

Details of the MLP predictive variables are included in the table below. The default parameter settings for training the MLP models are stored
in the `mlp_settings.yml <https://github.com/IMMM-SFA/tell/blob/main/tell/data/mlp_settings.yml>`_ file in */data* folder in the **tell** repository.
The hyperparameters for the **tell** MLP models (e.g., hidden layer sizes, maximum iterations, and validation fraction) were determined
using a grid search approach. Hyperparameters were allowed to vary across BAs. Default hyperparameters for each BA are
also included in the */data/models* folder in the **tell** repository.

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
    * - Hour of the day
      - Hour of the day in UTC
      - 00-23 UTC
    * - Day of the week
      - Is the day a weekday or weekend?
      - Weekdays (1) or Weekends (0)
    * - Federal holiday
      - Is the day a federal holiday?
      - Yes (1) or No (0)

In general, the **tell** empirical models work quite well. 76% (41/54) of the BAs have an R2 value greater than 0.75
while 83% (45/54) have a MAPE under 10%.

.. image:: _static/MLP_Summary_Statistics.png
   :width: 900
   :align: center

It's illustrative to look at the error metrics as a function of load. To do this, we calculate the mean hourly load
for each BA during the evaluation year and then plot the error statistics as a function of that mean load. Analyzing
the data in this way demonstrates that the BAs with the poorly performing empirical models are almost universally the
smaller BAs. The largest BAs, which are critically important for the overall demand on the grid, generally perform quite
well. Of the 10 BAs with the largest mean demand, 9/10 have a MAPE value under 5% and an R2 value greater than 0.85.
Conversely, of the 10 worst performing BAs (judged by their MAPE value), 7/10 have an average hourly load less than
1700 MWh.

.. image:: _static/MLP_Summary_Statistics_vs_Load.png
   :width: 900
   :align: center

Because the empirical models that underpin **tell** are so critically important we created a separate analysis notebook
where users can explore the model's performance characteristics collectively and for individual BAs. The MLP calibration
and evaluation notebook can be found
`here <https://github.com/IMMM-SFA/tell/blob/main/notebooks/tell_mlp_calibration_evaluation.ipynb>`_.


Incorporating Detailed Sectoral Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By design **tell** projects future time-series of the *total* hourly load at different spatial scales. These *total* loads
are responsive to variations in population and climate. It is important to note that **tell** does not resolve the
load profiles for individual sectors of the electric industry (e.g., residential, commercial, industrial, and commercial).
However, the model is designed so that it can be modified to reflect changes in these individual sectors in a relatively
straightforward way. We know that technologies in each of these sectors are currently and are likely to continue to evolve
quickly. For example, the rapid penetration of rooftop solar will modify future grid-scale electricity demand from residential
customers. Similarly, widespread adoption of electric vehicles will impact the magnitude and shape of the load profiles in the
residential, commercial, and transportation sectors. In order to reflect technology change in a given sector you need a detailed
understanding of that sector as well as an ability to simulate future changes due to specific technologies.

While **tell** was not designed for this level of detail, other detailed sectoral models are. We built **tell** to incorporate
technological changes by partnering with these detailed sectoral models. The figure below shows how this might work conceptually.
The top row reflects information that might come out of a detailed residential energy model. In panel (a) we show the diurnal
load profiles for residential customers in a given region. The load profile reflects a typical springtime load profile
in residential buildings. Now imagine that you wanted to simulate the impact of widespread rooftop solar adoption within that
region. Panel (b) shows the potential solar energy supply simulated by the detailed model. The solar energy curve follows a typical
sinusoidal pattern that peaks at solar noon. Finally, panel (c) shows the impact of rooftop solar on the residential demand profile.

.. image:: _static/Load_Perturbation_Incorporation_Example.png
   :width: 900
   :align: center

**tell** can take the output of the detailed residential buildings sector model and use it to modify the time-series
of *total* load that the model projects. The way to do this is to take the difference values produced by the detailed sectoral
model (i.e., the difference between the base and modified residential load profiles) and add those perturbations directly on top
of the *total* load time-series produced by **tell**. Panel (d) shows how this would play out in **tell**. The black line represents
the **tell** hourly *total* load time-series before the intervention and the red line shows the *total* load time-series after the
rooftop solar difference values from the residential model were added.

This approach means that **tell** doesn't need to know anything about the residential energy sector or the fraction of the total
load it represents. All **tell** cares about is how the intervention you want to explore will translate into changes in the sectoral
load time-series. Note that in order to do this the detailed sectoral model needs to produce output at at least one of the spatial
scales in **tell** (i.e., counties, states, or BAs). This approach allows users of detailed sectoral models to explore how specific
interventions will impact future demands at the grid-scale without having to have complementary sectoral models of all other sectors.
Finally, if the detailed sectoral model projects changes in the load shape but doesn't resolve the magnitude at a given spatial scale,
it should be possible to use year-over-year changes from the GCAM-USA sectoral models to scale the load shape changes before they are
passed on to **tell**.


Scenarios
~~~~~~~~~
**tell** is designed to work in conjunction with the U.S. version of the Global Change Analysis Model (GCAM-USA)
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


Outputs
-------
**tell** produces four types of output files. Each type of output is written out as a .csv file or series of .csv files in ``tell_data/outputs/tell_output/scenario_name``.
Each type of output file can be suppressed by commenting out the relevant output function in ``execute_forward.py``. Missing values in each output file are
coded as -9999. All times are in UTC.


State Summary Data
~~~~~~~~~~~~~~~~~~
This output file gives the annual total loads for each of the 48 states in the CONUS as well as the District of Columbia. It also contains the scaling factor for
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
      - FIPS code of the state
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
      - FIPS code of the state
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


County Hourly Load Data
~~~~~~~~~~~~~~~~~~~~~~~
This collection of output files gives the hourly time-series of total loads for each county in the CONUS and the District of Columbia.
These output files are stored in a subdirectory of the output directory named ``County_Level_Data``. Note that since it takes a while to
write out the county-level output data this output is optional. To output county-level load projections just set the ``save_county_data``
flag to True when calling the ``tell.execute_forward`` function.

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
      - FIPS code of the county
      - NA
    * - State_Name
      - Name of the state the county is in
      - NA
    * - State_FIPS
      - FIPS code of the state
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