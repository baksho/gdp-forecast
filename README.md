## gdp_forecast
This model is aimed to forecast the GDP of a country using machine learning algorithms.

***This project is not final. Updates are in progress.***

#### Aim
In this simple GDP forecasting model using a machine learning approach with Python, we aim to predict the GDP of a country in future from historical GDP data and various other basic economic indicators.
The model is pretty simple and use Random Forest Regressor, a robust esemble learning method.

#### Economic Indicators
Following economic indicators were used to forecast the GDP in future along with historical GDP data:
- Interest rate
- Inflation rate
- Unemployment rate

#### Future Work
As of now, the model is pretty simple and uses synthetica data (generated in Python) for testing the functionality of code. Following works are planned to carry out in future:
1. Use real time sources for historical GDP data and basic economic indicators, e.g.:
  - [World Bank](https://data.worldbank.org/)
  - [OECD](https://data-explorer.oecd.org/)
  - [IMF](https://data.imf.org/?sk=388dfa60-1d26-4ade-b505-a05a558d9a42)
  - [US Bureau of Economic Analysis](https://www.bea.gov/)

2. Use more complex models like Deep Neural Network.
3. Use Recurrent Neural Network (RNN) for temporal and sequential data.
4. Explore how we can tackle sudden spike / drop in the historical data due to unforeseen situations (war-time situation, economic recession, medical emergency like COVID-19), which impacts the GDP to a great extent.

#### Reference Academic Papers
