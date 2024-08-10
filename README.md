
# Phase diagram and Quantum Phase transition.
This response is an example of using the VGG Deep learning model to draw the phase diagram in condensed matter. The train/test data is a 2-dimensional spectral function (SF) in  $` A(\omega,k)`$
where they are distinguished by the existence of the gapped and gapless spectral features. The zero or very small values between the lower and upper bands will be called the gapped phase. On the other hand, where the upper band and lower band touch each other, we call the gapless phase. Tuning parameters such as temperature, coupling constant, order parameter, and chemical potential usually cause the phase transition. </br>

In this response, I will give just only 2 examples: </br>
<ul>
<li> The quantum phase diagram or, simply speaking, the phase transition at the zero temperature. In this case, the phase diagram can be completed by varying 2 parameters, so-called chemical potential and coupling constant.</li>
<li> the finite temperature phase diagram. In this case, the temperature is also varied so that the phase diagram appears as a three-dimensional object.</li>
</ul>

However, the Deep learning model is still valid for both cases.

# Training Data

In this example, I generate the spectral function by solving a system of differential equations. See my college paper [Classification of Mott Gap](https://doi.org/10.48550/arXiv.2404.10412). I also provided the Julia
code to generate the spectral function. See. 

The train data will be divided into 2 classes: gapped and gapless. The dataset is collected in CSV (data frame) format. I uploaded all the data here: [Drive](https://drive.google.com/drive/folders/18zn7zBnz3JQzz35ODgLYF7AWuLrYS50E?usp=drive_link)

# The Classification Model.

We will feed the model data by diverging gapped and gapless spectral functions. The pseudo gap occurs when there is a small value between the upper and lower bands but not low enough to be decided as gapped or gapless. In the practical data, we will use the model to draw the phase diagram of course, containing these phases. Therefore, I will use a softmax activation function, which will give the probability for each class. From this fact, I can assume that when the absolute value of the difference of probability is small enough, let's say it is a pseudo-gap phase.
