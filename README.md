# BCI Simulator

BCI simulator is a lightweight & simple simulator for closed-loop brain-computer interfaces (BCIs). 
It is designed for researchers to quickly test out decoder algorithms and get a "feel" for how they might work in a closed-loop setting.
It may also be helpful to teach students about the implementation of closed-loop, real-time BCIs.


There are three main components to the simulator:
- **Task**: the task is the environment that the user interacts with. The default task is a 2D cursor task, where the user has to move the cursor to hit a target. 
- **Neural Generator**: the neural generator creates artificial neural data using kinematics (position + velocity) as input. 
We currently implement a simple log-linear tuning model, but plan to add more complex models in the future.
- **Input:** inputs include the mouse or a decoder (see pipelines below). 
The decoder is the algorithm that takes in neural data and predicts user intentions (movements). 
We give examples for RNN and ridge regression decoders.

Pipeline using a mouse as input:
- mouse -> task

Pipeline using a decoder as input:
- mouse (intended movement) -> neural generator -> decoder (predicted movement) -> task


This project is largely inspired by other closed loop simulators, 
including the [AE Studio Neural Data Simulator](https://github.com/agencyenterprise/neural-data-simulator),
[Willet et al. 2019](https://www.nature.com/articles/s41598-019-44166-7), 
[Cunningham et al. 2011](https://journals.physiology.org/doi/full/10.1152/jn.00503.2010),
and [BRAND/Ali et al. 2023](https://www.biorxiv.org/content/10.1101/2023.08.08.552473v1.full).


## Real-Time Clocking/Timing Notes

- For now, we don't incorporate any precise clocking/timing in the desire for simplicity. 
The update rate is determined by the `clock.tick(FPS)` command within the task loop (a pygame command).

- We simulate neural data at the bin level, rather than the spike level. 
For example, we use 20ms timesteps, so that the neural data generated is the average firing rate over a 20ms bin.
This minimizes the amount of data that needs to be generated and processed.

## Parameters to explore
A few system/decoder parameters that may be interesting to explore:
- **Time History**: how many previous timesteps of neural data to use as input to the decoder. 
Increased time history typically improves offline accuracy, but usually harms closed-loop performance.
- **Neural Noise & Number of Channels**: increasing neural noise adds more variability to the generated neural data,
and increasing the number of channels increases the amount of information available to the decoder.
- **Decoder Architecture**: many groups have turned toward deep learning approaches for BCI decoders



## Installation
Clone the repository:
```
git clone https://github.com/jtcostello/bmisimulator.git
```

Create a conda environment and activate:
```
conda create -n bmisimulator python=3.9
conda activate bmisimulator
```

Install packages:
```
pip install -r requirements.txt
```

## Usage

Run the 2d cursor task, using the mouse as input:
```
   python main_run_task.py -i mouse -t cursor
```

Train a decoder (100 channels, neural noise of 0.3):
```
   python main_train_decoder.py -c 100 -n 0.3 -d dataset_20230929_2158.pkl 
```

Run the 2d cursor task, using a decoder as input:
```
   python main.py -i mouse -t cursor -d decoder_rnn_dataset_20230929_2158.pkl
```


## Relevant research papers:
(not at all an exhaustive list, just a few relevant papers)

A key challenge in the design of real-time BCI algorithms is how the offline accuracy of the decoder does not 
directly indicate how well it will perform in a closed-loop setting. 

**Importance of incorporating closed-loop feedback in BCI simulations:**
- Chase et al 2009
- Koyama et al 2010
- Merel et al 2016
- Zhang and Chase 2018
- [Cunningham et al. 2011](https://journals.physiology.org/doi/full/10.1152/jn.00503.2010)
- [Willet et al. 2019](https://www.nature.com/articles/s41598-019-44166-7)

**Simulating Neural Data:**
- Liang et al. 2020, Deep Learning Neural Encoders for Motor Cortex
- Awasthi et al. 2022, Validation of a non-invasive, real-time, human-in-the-loop model of intracortical brain-computer interfaces


