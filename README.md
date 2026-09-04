# Soho Instant-Delivery Agent-Based Model

## Overview

This repository contains the source code for the MSc dissertation
“Algorithmic Pressure and Micro-spatial Conflicts: An Agent-Based
Model of Instant Delivery Flows in Soho, London”.

The model examines how platform time pressure, rider behaviour and
street-level spatial constraints interact to generate pavement use,
risky intersection crossings and spatially concentrated exposure.

The model is explanatory and exploratory. Its outputs represent
relative responses within the model and should not be interpreted as
predictions of real-world collision or violation rates.

## Software

- Python: [version]
- Mesa: [version]
- GeoPandas: [version]
- OSMnx: [version]
- NetworkX: [version]
- Shapely: [version]
- pandas: [version]
- NumPy: [version]
- Matplotlib: [version]

## Model configuration

- Study area: Soho-centred modelling extent, London
- Spatial radius: 1,900 m
- Simulation period: 17:00–20:00
- Time step: 5 seconds
- Number of riders: 300
- Number of pedestrians: 600
- Formal repetitions: five fixed random seeds

## Repository structure

- `model/`: model agents, scheduling and behavioural mechanisms
- `data/`: data documentation and preprocessing information
- `outputs/example/`: example model outputs
- `requirements.txt`: required Python packages

## Installation

```bash
git clone [repository URL]
cd soho-instant-delivery-abm
python3 -m pip install -r requirements.txt
