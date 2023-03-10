# Прогнозирование заказов такси


## Описание проекта

Компания `«***»` собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Построим модель для такого предсказания.

Значение метрики **RMSE** на тестовой выборке должно быть не больше **`48`**.

Нам нужно:

* Загрузить данные и выполнить их ресемплирование по одному часу;
* Проанализировать данные;
* Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных;

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
```

---

## Итоги исследования

* Данные были загружены и просмотрены введенной функцией;
* Исправили тип данных с **str** строкового на **datetime64** при помощи аргумента `parse_dates=[index]`, чтобы было в дальнейшем удобнее работать с временным рядом;
* При помощи аргумента `index_col=[index]` установили индекс таблицы равным столбцу **datetime**;
* Произвели **ресемплирование** по **1 часу**;
* Разложили временной ряд на `тренд`, `сезонность` и `остаток` при помощи библиотеки `statsmodels` с функцией `seasonal_decompose()`;
* Проанализировали активность заказов такси по нему;
* Были добавлены новые признаки для обучения моделей;
* Данные были разделены в соответствии условий (**10% тестовой выборки** от исходных данных);
* Была добавлена функция **rmse**, как расчет метрики **RMSE**;
* Также была введена функция **best_model**, в которую была передана кастомная функция **rmse** как `score`. Также благодаря ей были рассчитаны **RMSE** моделей и лучшие гиперпараметры для них при помощи **GridSearchCv**;
* Были приведены **три модели** в качестве прогнозирования кол-во заказов такси на следующий час: `LinearRegression`, `CatBoostRegressor` и `LGMBRegressor`;
* В результате анализа и финальной проверки на тестовой выборке, выявили модели для компании «Чётенькое такси»: 
  
  * **CatBoostRegressor `(~ 46.8)`** и **LGMBRegressor `(~ 47.3)`**
