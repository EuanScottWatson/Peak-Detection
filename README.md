# Peaks in Time-Series Stocks
Simple peak detection algorithm I implemented following this research paper: https://www.researchgate.net/publication/228853276_Simple_Algorithms_for_Peak_Detection_in_Time-Series. <br>

# How To Run
Install all the requirements from the `requirements.txt` file then run using one of:
```
python model.py
python model.py <ticker symbol>
python model.py <ticker symbol> <start-date>
python model.py <ticker symbol> <start-date> <end-date>
```

By default, it will display the peaks in Tesla's stock from 1st of Jan 2020 to 1st of Jan 2021.