# Исследование надежности заемщиков


## Описание проекта

Была поставлена задача заказчиком, которым является `банк`. 

Нужно проанализировать семейное положение и количество детей клиента на факт погашения кредита в срок. 

Даны данные статистики о платёжеспособности клиентов. Все наши результаты исследования будут учтены при построении модели кредитного скоринга — специальной системы, которая оценивает способность потенциального заёмщика вернуть кредит банку.

---

## Стек/инструменты/библиотеки

```python
import pandas as pd
```

---

## Итоги исследования

* Люди с доходом выше среднего более безопасные клиенты;
* Люди в браке более отвественные плательщики;
* Кол-во детей напрямую отражает выплату в срок кредита;
* Выдача кредита с целью `'операции с недвижимостью'` является более безопасной в плане задержки срока сдачи;
* Цель кредита `'операции с автомобилем'` является более рискованной;
* Стоит рассматривать источник дохода заемщика, особенно обращая внимание на категорию `'сотрудник'`

