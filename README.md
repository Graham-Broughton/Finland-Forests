# DrivenData-The BioMassters

## Overview

The objective of [this competition](https://www.drivendata.org/competitions/99/biomass-estimation/) was to predict the yearly maximum aboveground biomass of certain areas of Finland using Sentinel 1 & 2 satellite images with LiDAR for ground truth.

## Challenges

Divising a method that incorporates all 12 time points.
This would make the dimensions batch size x time dimension x height x width x channels (b x 12 x h x w x 15)