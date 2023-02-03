# Прогнозирование оттока клиента банка



## Описание проекта

Из `«***»` стали уходить клиенты. Каждый месяц. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.

Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Нам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 

Стоит задача построить модель с предельно большим значением `F1`-меры. Был установлен минимум значения метрики - до `0.59`.

Дополнительно будем измерять `AUC-ROC`, сравнивая её значение с `F1`-мерой.

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle 
from sklearn.preprocessing import StandardScaler
```

---

## Итоги исследования

**`Лучшая модель прошла проверку на тестовой выборке` и получилось удолетворить условие задачи в достижении f1 метрики (`выше 0.59`)**
   * `Тестовая точность`:0.814
   * `Тестовая метрика f1`: 0.609
   * `Тестовая метрика AUC-ROC`: 0.768

  
   * `accuracy_score_best выявленной модели`: 0.829
   * `f1_best`: 0.631
   * `AUC-ROC best`: 0.781
