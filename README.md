[![License](https://img.shields.io/github/license/xsolla/data_fast_insights)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-blue)](https://img.shields.io/badge/Python-3.7%2B-blue)
[![Version](https://img.shields.io/badge/version-0.2.1.1-blue)](https://img.shields.io/badge/version-0.2.1.1-blue)

# Info
**Data Fast Insights** is a library for quickly finding insights on given data or testing hypotheses.  
Thus it can be used:
 - to test how any features from a given set (as well as their combinations) affect target variable 
 and **find those which require attention and change the most**  
 - to test if a defined hypothesis is true or not  

[What does the library calculate and how?](doc/CALCULATIONS_DESCRIPTION.md)

# Installation  
Installation instructions can be found [here](doc/INSTALL.md). 


# Examples  
- [Quickstart example in the Jupyter Notebook format](examples/housing_dataset_example.ipynb)
- **More verbose example on synthetic data** with comments on code and its output can be found 
[here](doc/VERBOSE_EXAMPLE.md).  
- [Other examples](examples/) (IPython notebooks and python scripts)  

# Other features
Features not used in examples are described [here](doc/OTHER_FEATURES.md)

# Changelog
You can find the version history [here](CHANGELOG.md).

# To be done  

#### Backlog (top priority features)
- Renaming, deleting, adding features inside the model
- Improving overall report interpretability
- Adding new metrics
- Adding estimated effect of changing segments to report (now it's in a distinct dataframe only)
- Adding tests
- Describing and improving additional and experimental features (such as split-apply-combine experiment type)
- Improving support of combined features in plotting 
- Automatic result from just SQL that generates features and target