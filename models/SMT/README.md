# Introduction

Among the various models we worked on, the most lightweight model is SMT (Stacked Machine-learnings with TabNet). This model involves generating metadata through various machine learning models and using TabNet, which is optimized for structured data, to classify anomalous events.

SMT has achieved an accuracy of 94.6% and an F1-score of 0.66. Although its performance is relatively lower compared to MLFC (Multi-Layer Fusion Classifier), it has the advantage of classifying anomalous events with decent performance through short training times and minimal resources.



# Installment

------

* First, Intall requirement file

```python
pip install -r requirements.txt
```

* Then, write your 3W dataset file path.

```
FILE_PATH = "WRITE YOUR PATH"
```

* Finally, run jupyter notebook file



# Architecture

Model architecture

![image-20230831162937657](https://github.com/lofootve/geo-con/assets/119025706/63951cb7-b739-429a-bdea-600535713237)