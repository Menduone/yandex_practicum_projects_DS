# Исследование объявлений о продаже квартир


## Описание проекта

Даны данные сервиса `Яндекc.Недвижимость` — архив объявлений о продаже квартир в `Санкт-Петербурге` и соседних населённых пунктах за несколько лет. 

Нужно научиться определять рыночную стоимость объектов недвижимости. 

Наша задача — установить параметры. Это позволит построить автоматизированную систему: она отследит аномалии и мошенническую деятельность.

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Итоги исследования

* Мы разобрали данные в таблице `data`
* **Посчитали и добавили в таблицу**: (`цену квадратного метра`, `день недели`, `месяц` и `год публикации объявления`, `этаж квартиры` (первый, `который оказался дешевле, чем остальные`, последний, другой), `соотношение жилой и общей площади`, а также `отношение площади кухни к общей`;
* Было обнаружено, что **распределение параметров квартир** (площадь, цена, число комнат, высота потолков) близки к **Пуассоновскому** (число событий в единицу времени, если они в среднем происходят с измеренной частотой);
* Изучили **время продажи квартиры**. То, что идет после **`500`** дней - считается необычно долгой продажей. А то что до первого квартиля (**`45`** дней) - очень быстро. Имеется достаточно большой разброс продажи квартир - **45-232 дней**;
* Подметели, что на **стоимость** квартиры **больше** всего **влияют** параметры, такие как: `общая площадь`, `колличество комнат`, `удалённость от центра`;
* Мы выбрали **10** населенных пунктов с **наибольшим** числом объявлений. Самая **высокая** стоимость жилья `была обнаружена в Санкт-Петербурге`, а самая низкая в **Выборге**;
* **Создали** отдельную **выборку** по квартирам **Санкт-Петербурга `local_spb`** и посчитали **границы** его **центра** (они оказались в пределах **8 км**);
* Выделили **сегмент для центральних квартир Санкт-Петербурга**. Изучилили территорию этих квартир, в частности их параметры (`площадь, цена, число комнат, высота потолков`);
* Мы выявили, что квартиры **в центре** Санктр-Петербуига принимают **большие значения** по сравнению с квартирами по всему городу;
* Наглядно показали то, что был замечен **спад цен на недвижимость** с **2014 на 2015** года (особенно для квартир в **центральном районе Санкт-Петербурга**);
* Рынок **нежвижимости** стал немного оправляться после спада цен к **2019** году. Был **заметен подъем цен**
