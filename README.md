# Phase diagram and Quantum Phase transition.
This response is an example of using the VGG Deep learning model to draw the phase diagram in condensed matter. The train/test data is a 2-dimensional spectral function (SF) in  `A(\omega,k_x,k_y,k_z)`  (energy/momentum space)
where there is a gap between the lower and upper bands, it will be called the gapped phase. On the other hand, where the upper band and lower band touch each other, we call the gapless phase. Tuning parameters such as temperature, coupling constant, order parameter, and chemical potential usually cause the phase transition. </br>

In this response, I will give just only 2 examples: </br>
1) The quantum phase diagram or, simply speaking, the phase transition at the zero temperature. In this case, the phase diagram can be completed by varying 2 parameters, so-called chemical potential and coupling constant. <\br>
2) the finite temperature phase diagram. In this case, the temperature is also varied so that the phase diagram appears as a three-dimensional object.

However, the Deep learning model is still valid for both cases.

# 
