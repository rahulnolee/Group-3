# Meachine Learning Prediction Website - Rent and Placement

This is a machine learning web application that provides predictions for two datasets: housing rent prediction and campus placement prediction.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

## Overview

The project consists of two main components:
- **Data Collection:** This module collects and preprocesses data from two datasets: "House_Rent_Dataset.csv" and "Placement_Data_Full_Class.csv".
- **Data Processing:** This module preprocesses and analyzes the collected data, handles missing values, encodes categorical variables, and trains machine learning models for prediction.

## Technologies Used

- Python
- Flask
- pandas
- scikit-learn
- SQLite
- HTML/CSS
- Pickel

## Installation

1. Clone the repository to your local machine: git clone [https://github.com/your-username/ml-web-app.git](https://github.com/rahulnolee/Group-3.git)

2. Navigate to the project directory: cd ml-web-app

3. Install dependencies:pip install -r requirements.txt


## Usage

1. **Run the Data Processing module:**
- Navigate to the `dataprocessing` directory and run the Jupyter notebooks (`housing.ipynb` and `students.ipynb`) to preprocess and analyze the collected data.

2. **Run the Flask Application:**
- Navigate to the project root directory and run `app.py`:
  ```
  python app.py //in the terminal
  ```
- Access the web application at `http://localhost:5003`.

## Directory Structure
![image](https://github.com/rahulnolee/Group-3/assets/113876448/28d532c7-902a-463f-85c7-27885711e176)


## Purpose of This Project

The purpose of this project is to create a web application that provides predictive analysis for housing rent and campus placement based on the provided datasets. It aims to demonstrate the use of machine learning techniques in real-world scenarios and provide insights into data preprocessing, model training, and web application development.
