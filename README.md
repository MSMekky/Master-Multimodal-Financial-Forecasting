Multimodal Financial Forecasting

Overview

This project implements a multimodal financial forecasting system that integrates Google Trends, technical indicators, and deep learning (BiLSTM) to predict asset price movements and market behavior. It explores how collective attention improves forecasting accuracy, especially during crisis periods.

⸻

Features 

• Combines Google Trends, technical data, and price history 

• Deep learning models for time-series forecasting 

• Feature importance and correlation analysis

• Scenario testing (e.g., COVID-19, 2022 crash) 

• Interactive dashboard with visualizations

⸻

Project Structure 

• data/ – datasets used in the project 

• outputs/ – model results, predictions, and figures 

• thesis_pipeline.py – main training and data processing pipeline

• thesis_results.py – evaluation, backtesting, and analysis 

• thesis_visualization.py – visualization and dashboard script

• README.md – documentation 

• requirements.txt – Python dependencies

⸻

Usage
1. Train and evaluate models by running:
   
 python thesis_pipeline.py

 python thesis_results.py
 
2. Launch the interactive dashboard:
   
 python thesis_visualization.py --dash

⸻

Installation

Clone the repository and install dependencies:

git clone (https://github.com/MSMekky/Master-Multimodal-Financial-Forecasting) cd Multimodal-Financial-Forecasting pip install -r requirements.txt

Citation


Author, M. (2025). Multimodal Financial Forecasting: Integrating Google Trends, Technical Indicators, and Deep Learning. GISMA University of Applied Sciences.



