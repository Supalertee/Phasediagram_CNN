# Phase diagram and Quantum Phase transition.
This response is an example of using the VGG Deep learning model to draw the phase diagram in condensed matter. The train/test data is a 2-dimensional spectral function (SF) in  $` A(\omega,k)`$
where there is a gap between the lower and upper bands, it will be called the gapped phase. On the other hand, where the upper band and lower band touch each other, we call the gapless phase. Tuning parameters such as temperature, coupling constant, order parameter, and chemical potential usually cause the phase transition. </br>

In this response, I will give just only 2 examples: </br>
1) The quantum phase diagram or, simply speaking, the phase transition at the zero temperature. In this case, the phase diagram can be completed by varying 2 parameters, so-called chemical potential and coupling constant. </br>
2) the finite temperature phase diagram. In this case, the temperature is also varied so that the phase diagram appears as a three-dimensional object.

However, the Deep learning model is still valid for both cases.

# Train Data

In this example, I generate the spectral function by solving a system of differential equations. See my college paper [Classification of Mott Gap] (https://doi.org/10.48550/arXiv.2404.10412). I also provided the Julia
code to generate the spectral function. See. 

The train data will be divided into 2 classes: gapped and gapless. The dataset is collected in CSV (data frame) format.
