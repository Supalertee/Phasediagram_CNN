
# Phase diagram and Quantum Phase transition.
This repository is an example of using the VGG Deep learning model to draw the phase diagram in condensed matter. The train/test data is a 2-dimensional spectral function (SF) in  $` A(\omega,k)`$
where the existence of the gap and gapless spectral features distinguishes them. The zero or very small values between the lower and upper bands will be called the gapped phase. On the other hand, where the upper band and lower band touch each other, we call the gapless phase. Tuning parameters such as temperature, coupling constant, order parameter, and chemical potential usually cause the phase transition. </br>

For the brief theoretical background and conclusion of this project, I recommend to visit  <a href = "https://sukrakarn-sci.com/DeepGap.php"> My website</a>

In this response, I will give just only 2 examples: </br>
<ul>
<li> The quantum phase diagram or, simply speaking, the phase transition at the zero temperature. In this case, the phase diagram can be completed by varying 2 parameters, so-called chemical potential and coupling constant.</li>
<li> the finite temperature phase diagram. In this case, the temperature also varies so that the phase diagram appears as a three-dimensional object.</li>
</ul>

However, the Deep learning model is still valid for both cases.

# Training Data
<ul>  
  <li>For theoretical fundamentals to understand the Gapped/Gappless phase and how to get the Green's function and spectral function, I recommend my colleague's paper 
    <a href = "https://doi.org/10.48550/arXiv.2404.10412">Classification of Mott Gap</a></li>
  <li> For the fast-solving differential equations for getting spectral functions. See <a href = "#">Solving Differential Equations by Julia</a></li>
  <li> I also provided the dataset for training the model, divided into 2 classes: gapped and gapless. The dataset is collected in CSV (data frame) format. See  <a href = "https://drive.google.com/drive/folders/18zn7zBnz3JQzz35ODgLYF7AWuLrYS50E?usp=drive_link">Dataset from my Google drive</a></li>
</ul>


# The Classification Model.

<ul>
  <li> Size of the input is (241,241,1) </li>
  <li> Convolution 2D with ReLU activation function </li>
  <li> Dropout can also be used for adding the regulator </li>
  <li> Softmax activation function will be used in the final layer even though there are just 2 classes in the training data. The reason is given below </li>
</ul>
We will feed the model data by diverging gapped and gapless spectral functions with size (241,241,1). However, the CSV files in my Google Drive are flattened to $ 241^2 $. So, if you use my data, you should reshape them. </br>

Moreover, the intermediate phase, the so-called pseudo gap, occurs when there is a small value between the upper and lower bands but not low enough to be decided as gapped or gapless. In the practical data, we will use the model to draw the phase diagram containing these phases. Therefore, I will use the **$$\text{\color{red}{softmax activation function}}$$**, which will give the $$\text{\color{lightgreen}{probability for each class}}$$. From this fact, I can assume that when the absolute value of the probability difference is small enough, let's say it is a pseudo-gap phase.
