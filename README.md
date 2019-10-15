# FRETboard: supervise your FRET detection algorithm
#### [Carlos de Lannoy](https://www.vcard.wur.nl/Views/Profile/View.aspx?id=77824), [Dick de Ridder](https://www.vcard.wur.nl/Views/Profile/View.aspx?id=56806&ln=eng)

FRETboard helps you train algorithms for the detection of Förster resonance energy transfer events in a 
(semi-)supervised manner.

## Running
FRETboard is available as web application on [Heroku](https://fret-board.herokuapp.com/), or can be installed on your own 
system using pip:

```
pip install git+https://github.com/cvdelannoy/poreTally.git
```
FRETboard is then started from the command line as:

```
FRETboard 
```
A session on a random free port will start automatically.

## Usage
Training an algorithm using FRETboard is easy; just follow the steps in the left column of your screen:
![GUI example](FRETboard_example_screen.png)

#### 1. Load
Pick an algorithm from the drop-down menu. For now two types are available:
- The vanilla hidden Markov model (HMM) - a simple fully connected HMM - should do for low noise data.
- The boundary-aware HMM tries to improve state detection by adding extra states at the bounds between the states you 
want to recognize, which works better in noisy traces.

You may now load model parameters of a previous time you used FRETboard if you have them, otherwise you will start 
with a fresh model. Then load your data with the Data button.

#### 2. Teach
After loading your data you are presented with a random example trace. Caught an error? Slide the 'Change selection' 
slider to the state you would like to introduce or expand and click-drag over the trace in your screen. You have now 
adapted the labeling of that trace. Once you are satisfied with the current trace, click 'Train' to retrain 
the algorithm using the modifications you just made as a guideline.

A few more options are available to you in this stage:
- Number of states: changes how many states the HMM will try to fit. Note that changing this value will reset your model!
- Influence supervision: choose the weight of supervised examples during training. 1 denotes fully supervised training,
0 means that supervised examples play no role at all.
- Buffer: if you're training a boundary-aware HMM (and possibly other algorithms in the future), choose how many data points the boundary states should cover. As a
rule of thumb, check how many measurements it takes to transit from one state to the next and pick that as a value here.
- Show traces with states: deselecting a certain state omits traces containing that state from further supervision.
- Delete: delete the current trace.
 
#### 3. Save
You may now download the classified traces on your machine using the 'Data' button. Produce a 
[Report](FRETboard_example_report.html) (download and view in browser) to see the
model parameters generated, along with some handy summary statistics and graphs or download the model to 
quickly classify your data next time using the same parameters. Deselecting certain states here omits traces containing
that state from your save file.

## Writing new algorithms
If you would like to introduce a new (semi-)supervised algorithm to FRETboard, you can do so easily; follow the 
instructions [template](model_template.py) and everything should work accordingly. Do consider making a pull request 
if you think your implementation may be useful to others! Of course, contributors will be fairly referred to. 

---
with &hearts; from Wageningen University