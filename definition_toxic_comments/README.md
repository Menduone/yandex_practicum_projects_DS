# Определение токсичности комментариев


## Описание проекта


Интернет-магазин `«***»` запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 

Обучим модель классифицировать комментарии на позитивные и негативные. В нашем распоряжении набор данных с разметкой о токсичности правок.

Построим модель со значением метрики качества `F1` не меньше **0.75**. 

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
import matplotlib.pyplot as plt
import sys
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import shuffle
!{sys.executable} -m pip install spacy
!{sys.executable} -m spacy download en_core_web_sm
```

---

## Итоги исследования

* Были загружены и разобраны данные;
* Была произведена лемматизация с помощью библиотеки **spacy** модели **en_core_web_sm**;
* Также была произведена очистка от лишних символов;
* Избавились от стоп слов;
* Была произведена векторизация при помощи **TfidfVectorizer()**;
* С помощью техники **GridSearchCV** обучали разные модели на разных выборках и подбирали гиперпараметры: `LogisticRegression`, `LightGBMClassifier` и `CatBoostClassifier`;
* В ходе анализа, были подобраны лучшие гиперпараметры моделям с разными выборками для метрики **f1**;
* Были обнаружены лучшие значения метрики **f1** на `upsampled` выборках: `LogisticRegression (upsampled)`, `LightGBMClassifier (upsampled)` и `CatBoostClassifier (upsampled)`;
* Лучшие модели с **f1** значением были проверены на тестовой выборке;
* Из трех выявленных моделей нужное значение **f1** показала только модель `LightGBMClassifier (upsampled)`. Она была проверена на адекватность. **Интернет-магазин `«Викишоп»` может использовать эту модель для поиска токсичных комментариев и отправлять их на модерацию**
