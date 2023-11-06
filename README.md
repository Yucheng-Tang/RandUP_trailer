# Sampling based reachability analysis for a tractor-trailer system

## About
Code for calculating the reachable set of a tractor-trailer system and representing it with convex hull. 

RandUP is an easy-to-implement reachability analysis algorithm. It consists of 1) sampling inputs, 2) propagating them through the reachability map, and 3) taking the epsilon-padded convex hull of the outputs. 

## Setup

This code was tested with Python 3.10.8. 

All dependencies (i.e., numpy, scipy, and matplotlib) can be installed by running 
``
  pip install -r requirements.txt
``

## TODO list
- [ ] add epsilon-padding function
- [ ] add intersection area calculation method (does Cal_area_2poly function work?)
- [ ] which one makes more sense (ys or ys_random?)
- [ ] train a model for intersection area calculation
