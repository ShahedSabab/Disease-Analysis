# Disease-Analysis
The objective is to detect patient with Parkinson's Disease (PD) from the voice samples. The training data includes voice measurements such as average, maximum, and minimum vocal fundemental frequency, several measureres of varation in fundemental frequency, variation in amplitudes, ratio of noise to tonal components, signal fractal scaling exponent, nonlinear measures of fundamental frequency variation. From these measurements, feature-selection (filter and wrapper method) is performed. After selecting important features different classifciaton algorithm is applied. The best performance (88% accuracy) is achieved using Random Forest with sfs-forward(sequential feature selector).

# Dataset:<br />
The dataset is collected from: https://archive.ics.uci.edu/ml/datasets/Parkinsons<br>
Parkinson.csv <br />
Numeber of Samples: 195<br />
Number of Features: 23 (voice measures)<br />

# Applied Techniques:<br />
Classifier:<br/>
• Random Forest<br />
• Logistic Regression <br />
• SVC<br />
• Naive Bayes <br />
• KNN<br />

Filter Method: <br />
• chi2<br />
• ANOVA F test<br />
• Mutual Information (MI)<br />

Wrapper Method: <br />
• Recursive Feature Elimination<br />
• Sequential Feature Selector (sfs-forward & sfs-backward)<br />
• Exhaustive Feature Selector<br />

# Output:<br />
Check .csv files for detailed output<br />

# Performance:<br />
![Model Accuracy using Filter Method](modelAccuracy_filterMethod.png)
![Model Accuracy using Wrapepr Method](modelAccuracy_wrapperMethod.png)

