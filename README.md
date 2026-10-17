# HOW TO PROPERLY USE OUR CODE (MANUAL SECTION, LLM SECTION NOT INCLUDED)

Make sure to set up the environment before starting

    uv sync

## 1\. Data Acquisition

The code for this task is in "ift6758\notebooks\raw_data_acquisition.ipynb".

You can download data from NHL play-by-play API for a particular season by creating 
an object from class "NHLyear" with parameters for the year the season starts and 
the path to the folder you want to save the data.

Here's an example for season:

``` python
# Example of the class being used:
yr2020 = NHLyear(2020, "data/raw")  # Creation of the object
yr2020.download_all()  # downloads all the games found
```

Please make sure you download all the data from seasons 2016-2017 to 2022-2023 
before continuing.

A more detailed turorial for the data acquisition is available in our blog post
"2026-10-19-milestone-1.md".

In our implementation, the json files are saved in "ift6758\notebooks\data\raw".

## 2\. Interactive Debugging Tool

The code for this task is in "ift6758\notebooks\ipywidgets.ipynb".

Run each cell in order. When the ipywidgets are running, you can select the season, 
match type, game id and event id of your choice. The code will return information 
about the game, the coordinates of the event on the skating rink image and information 
about the event.

Run the last cell to close the ipywidgets.

## 3\. Tidy Data

The code for this task is in "ift6758\features\tidy.py"

You can use the function "json2tidy" in a notebook or python file by importing it:

``` python
from ift6758.features.tidy import json2tidy
```

Then, to convert a raw NHL play-by-play JSON data into a tidy DataFrame, you parse
the path of the json file containing the data.

For example:

``` python
json2tidy("ift6758/notebooks/data/raw/2022/2022020001.json")
```

## 4\. Simple Visualisations

The code for the figure of the first question is in "ift6758\visualizations\shot_style.ipynb"

The code for the figures of the second question is in "ift6758\visualizations\distance_goal.ipynb"

The code for the figure of the third question is in "ift6758\visualizations\goal_percentage.ipynb"

For this part, you just need to run the cells in order.

## 5\. Advanced Visualisations

The code for this task is in "ift6758\visualizations\shot_maps.ipynb"