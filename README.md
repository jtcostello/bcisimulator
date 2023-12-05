# BCI Simulator

BCI simulator is a lightweight simulator for closed-loop brain-computer interfaces (BCIs), with the goal of simplicity 
and being easily modifiable. It is designed for researchers to quickly test out decoder algorithms and get a "feel" for 
how they might work in a closed-loop setting. It may also be helpful to teach students about the implementation of closed-loop, real-time BCIs.

[//]: # (![cursor simulator]&#40;docs/img/cursortask.png&#41;)
<p align="center">
  <img src="docs/img/cursortask.png" alt="cursor simulator" height="400"/>
</p>

There are three main components to the simulator:
- **Task**: the task is the environment that the user interacts with. The default is a 2D cursor task, where the user has to move the cursor to hit a target. 
- **Neural Generator**: the neural generator creates artificial neural data using kinematics (position + velocity) as input. 
We currently implement a simple log-linear tuning model, but plan to add more complex models in the future.
- **Input:** inputs include the mouse, a decoder (see pipelines below), or any other input devices. 
The decoder is the algorithm that takes in neural data and predicts user intentions (movements). 
We give examples for RNN and ridge regression decoders.

Pipeline using a mouse as input:
- mouse -> task

Pipeline using a decoder as input:
- mouse (intended movement) -> neural generator -> decoder (predicted movement) -> task

This project is largely inspired by other closed loop simulators, 
including the [AE Studio Neural Data Simulator](https://github.com/agencyenterprise/neural-data-simulator),
[BRAND/Ali et al. 2023](https://www.biorxiv.org/content/10.1101/2023.08.08.552473v1.full),
[Willet et al. 2019](https://www.nature.com/articles/s41598-019-44166-7), 
and [Cunningham et al. 2011](https://journals.physiology.org/doi/full/10.1152/jn.00503.2010).


## Installation
Clone the repository:
```
git clone https://github.com/jtcostello/bcisimulator.git
```

Create a conda environment and activate:
```
conda create -n bcisimulator python=3.9
conda activate bcisimulator
```

Install packages:
```
pip install -r requirements.txt
```

## Usage

### 1. Collect movement data

Run the 2d cursor task on random targets, using the mouse as input:
```
   python main_run_task.py -t cursor
```

### 2. Train a decoder (this also creates a neural data simulator)

Train an RNN decoder (100 channels, neural noise of 0.1, trained for 30 epochs):
```
   python main_train_decoder.py -c 100 -n 0.1 --epochs 30 -d dataset_20231012_250sec_random.pkl -o rnndecoder1
```

Train a ridge regression decoder with 5 history bins (100 channels, neural noise of 0.1):
```
   python main_train_decoder.py -c 100 -n 0.1 -seq_len 5 -d dataset_20231012_250sec_random.pkl -o ridgedecoder1
```

### 3. Test the decoder in closed-loop (simluating neural data in real time)

Run the 2d cursor task on center-out targets, using a decoder as input:
```
   python main_run_task.py -t cursor -d rnndecoder1 -tt centerout
```

View all the available command line arguments:
```
   python main_run_task.py -h
   python main_train_decoder.py -h
```

## Simulator design notes
#### Real-Time Clocking/Timing
- For now, we don't incorporate any precise clocking/timing in the desire for simplicity. 
The update rate is determined by the `clock.tick(FPS)` command within the task loop (a pygame command).

- We simulate neural data at the bin level, rather than the spike/ephys level. 
For example, we use 20ms timesteps, so that the neural data generated is the average firing rate over a 20ms bin.
This minimizes the amount of data that needs to be generated and processed.

#### Neural Simulation
- The `LogLinUnitGenerator` simulates channels using a log-linear relationship, as in 
[Trucollo et al. 2008](https://www.jneurosci.org/content/28/5/1163.short). This makes
it easy to generate an arbitrary number of degrees-of-freedom (whereas cosine tuning typically is for 2D).
- A new neural simulator is created for each new decoder. This means if you train two decoders with the same neural
settings, they will have different neural data, and could have different performance. In the future we may add the 
ability to use the same neural generator for multiple decoders.
- Neural simulators approximate the random tuning we see in real neural data - they are by no means a perfect simulation.
- The `neural_noise` parameter sets the std of the noise added to the average firing rate.

## Parameters to explore
A few system/decoder parameters that may be interesting to explore:
- **Time History**: how many previous timesteps of neural data to use as input to the decoder. 
Increased time history typically improves offline accuracy, but usually harms closed-loop performance.
- **Neural Noise & Number of Channels**: increasing neural noise adds more variability to the generated neural data,
and increasing the number of channels increases the amount of information available to the decoder.
- **Decoder Architecture**: many groups have turned toward deep learning approaches for BCI decoders
- **Training Data**: How much training data is needed? Is random targets or center out targets better?


## Relevant research papers
(not at all an exhaustive list, just a few relevant papers)

A key challenge in the design of real-time BCI algorithms is how the offline accuracy of the decoder does not 
directly indicate how well it will perform in a closed-loop setting. 

**Importance of incorporating closed-loop feedback in BCI simulations:**
- Koyama et al. 2010
- Zhang and Chase 2018
- [Cunningham et al. 2011](https://journals.physiology.org/doi/full/10.1152/jn.00503.2010)
- [Willet et al. 2019](https://www.nature.com/articles/s41598-019-44166-7)

**Simulating Neural Data:**
- Liang et al. 2020, Deep Learning Neural Encoders for Motor Cortex
- Wen et al. 2021, Rapid adaptation of brain–computer interfaces to new neuronal ensembles or participants via generative modelling
- Awasthi et al. 2022, Validation of a non-invasive, real-time, human-in-the-loop model of intracortical brain-computer interfaces


## Todo / Future work
- Add a virtual hand environment, using Leap motion or Google mediapipe for hand tracking
