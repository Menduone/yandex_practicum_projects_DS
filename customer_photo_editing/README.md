# Определение возраста покупателей


## Описание проекта


Сетевой супермаркет `«***»` внедряет систему компьютерного зрения для обработки фотографий покупателей. Фотофиксация в прикассовой зоне поможет определять возраст клиентов, чтобы:

* Анализировать покупки и предлагать товары, которые могут заинтересовать покупателей этой возрастной группы;
* Контролировать добросовестность кассиров при продаже алкоголя.


Построим модель, которая по фотографии определит приблизительный возраст человека. В нашем распоряжении набор фотографий людей с указанием возраста

**ВАЖНО**

`Все вычисления были произведены на сервере с графической картой (GPU) Yandex Compute Cloud.` Данные были заргужены в проект как разметка **Markdown** с сервера.

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
```

---

## Итоги исследования

* В качестве лосс-функции использовали **mae**, это допустимо, но для задач регрессии более подходящей лосс-функцией является **mse**, с ней модели учатся быстрее и стабильнее, но в качестве экперимента было принято решение использовать первое;
* Данные были загружены и рассмотрены с помощью гистограмм и диаграммы размаха;
* Основная масса данных пришлась на возраст **20 - 40** лет;
* С помощью функций были загружены данные в загрузчик **ImageDataGenerator** и выгружены с помощью функции `flow_from_dataframe()`;
* При помощи аугментации - параметра горизонтального отражения `horizontal_flip=True`, была достигнута метка указаннная в задаче - **меньше 7**;
* Обученная сверточная сеть **ResNet50** показала метрику качества **MAE** равную **6.04181**, что является достаточно хорошим результатом;
* Этот показатель говорит о том, что ошибается модель примерно на **6 лет**, что является хорошим показателем
* Для модели:
    * Чтобы подбор шага был автоматическим, применяли алгоритм - оптимизатор **Adam**. Он подбирает различные параметры для разных нейронов, что также ускоряет обучение модели;
    * Размер батча был выбран небольшой, размером 32 фото. Чем больше изображений, тем лучше обучится нейронная сеть. Но в `GPU` не хватит на слишком много фото, поэтому был выбран такой размер батча. И были подобрано небольшое кол-во эпох - **10**, по истечению которых и был достигнут нужны результат;
    * Модель преодобучалась на **Imagenet** - инициализация весов. Чтобы результат стал лучше, модель была предобучена на **Imagenet**;
    * Исключили верхушку и завели после, чтобы адаптировать под нашу задачу при помощи `include_top=False`;
    * В последнем полносвязном слое был выведен один нейрон для предсказания числа - возраста
    
**Данная модель соответсвует полностью требованиям заказчика** **`«***»`**