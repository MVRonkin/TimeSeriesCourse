# Материалы текущей версии курса представлены в репозитории 
https://github.com/MVRonkin/TimeSeriesCourse/tree/main/2026 

<details>
Материалы состоят 

1.	из лекционных презентаций по следующим темам <ul>
<li>1.1.	Введение в предмет анализ временных рядов</li>
<li>1.2.	Модели ВР и постановка задач предсказания значений ВР</li>
<li>1.3.	Статистические свойства ВР</li>
<li>1.4.	Разложение ВР и наивные методы предсказания</li>
<li>1.5.	Методы на основе экспоненциального сглаживания</li>
<li>1.6.	Методы авторегрессии-скользящего среднего.</li>
<li>1.7.	Оценка качества предсказания ВР (метрики)</li>
<li>1.8.	Анализ невязок предсказаний ВР (остаточная часть)</li>
<li>1.9.	Оценка надежности предсказаний ВР</li>
<li>1.10.	Методы предсказания ВР с использованием машинного обучения.</li>
<li>1.11.	Методы работы с многомерными ВР</li>
<li>1.12.	Обнаружение аномалий Во ВР</li>
<li>1.13.	Глубокое обучение нейронных сетей в приложениях анализа ВР</li>
<li>1.14.	Особенности работы с индустриальными ВР</li>
<li>1.15.	Задача классификации ВР (доп. Раздел)</li>
<li>1.16.	Задачи непараметрического анализа слжных ВР (доп. Раздел).</li>
</ul>
2.	материалов практик в формате ipynb по следующим темам <ul>
<li>2.1.	Визуализация  и предварительный анализ ВР</li>
<li>2.2.	Фреймворки для решения задач предсказания ВР</li>
<li>2.3.	Методы авторегрессии-скользящего среднего</li>
<li>2.4.	Оценка качеста предсказаний ВР</li>
<li>2.5.	Продвинутые техники использования фреймворков в задачах предсказания ВР</li>
<li>2.6.	Использование методов машинного обучения в задачах предсказания ВР</li>
<li>2.7.	Использование методов глубоких нейронных сетей в задачах предсказания ВР</li>
<li>2.8.	Решение задач поиска аномалий во ВР</li>
<li>2.9.	Решение задач классификации ВР (доп. Раздел).</li>
</ul>

3.	итогового задания https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/README.md
</details>

# Анализ временных рядов

## Содержение курса
### Модуль 1 - Основные понятия предмета анализ временных рядов

| Занятие | Тема | Материалы |
|---------|------|-----------|
| Лекция 1 - 1/2 | Введение в АВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/01-1%20%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%20%D0%90%D0%92%D0%A0.pptx) |
| Лекция 1 - 2/2 | Модели ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/01-2%20%D0%9C%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%20%D0%92%D0%A0%20%D0%B8%20%D0%BF%D0%BE%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B8.pptx) |
| Лекция 2 - 1/2 | Статистические свойства ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/02-1%20%D0%A1%D1%82%D0%B0%D1%82%D0%B8%D1%81%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D1%81%D0%B2%D0%BE%D0%B9%D1%81%D1%82%D0%B2%D0%B0%20%D0%92%D0%A0%20.pptx) |
| Лекция 2 - 2/2 | Разложение ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/02-2%20%D0%A0%D0%B0%D0%B7%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%BD%D0%B0%D0%B8%D0%B2%D0%BD%D0%BE%D0%B5%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B5.pptx) |
| Практика 1 | EDA и визуализация ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/01%20-%20%D0%92%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BF%D1%80%D0%B5%D0%B4%D0%B2%D0%B0%D1%80%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%20%D1%81%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC%D0%B8%20.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/01%20-%20%D0%92%D0%B8%D0%B7%D1%83%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B8%20%D0%BF%D1%80%D0%B5%D0%B4%D0%B2%D0%B0%D1%80%D0%B8%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D0%B0%20%D1%81%20%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%BC%D0%B8%20.ipynb) |

### Модуль 2 - Предсказания временных рядов на основе статистических моделей

| Занятие | Тема | Материалы |
|---------|------|-----------|
| Лекция 3 - 1/2 | Методы экспоненциального сглаживания | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/03-1%20%D0%AD%D0%BA%D1%81%D0%BF%D0%BE%D0%BD%D0%B5%D0%BD%D1%86%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5%20%D1%81%D0%B3%D0%BB%D0%B0%D0%B6%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B8%20%D0%BF%D1%80%D0%BE%D1%84%D0%B5%D1%82.pptx) |
| Лекция 3 - 2/2 | Метрики АВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/03-2%20%D0%9C%D0%B5%D1%82%D1%80%D0%B8%D0%BA%D0%B8%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%20%D0%92%D0%A0.pptx) |
| Практика 2 | Базовые методы предсказания ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/02%20-%20StatForecast_Intro_TS_Analysis.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/02%20-%20StatForecast_Intro_TS_Analysis.ipynb) |
| Лекция 4 - 1/2 | ARIMA | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/04-1%20ARIMA.pptx) |
| Лекция 4 - 2/2 | Анализ остатков предсказания | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/04-2%20%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D0%BD%D0%B5%D0%B2%D1%8F%D0%B7%D0%BE%D0%BA%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B9%20%D0%92%D0%A0.pptx) |
| Практика 3 | ARIMA | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/03%20-%20StatForecast_DSModels(ARIMA).ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/03%20-%20StatForecast_DSModels(ARIMA).ipynb) |
| Практика 4 | Диагностика ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/04-NixtlaDiagnostics.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/04-NixtlaDiagnostics.ipynb) |

### Модуль 3 - Использование машинного обучения в задачах анализа временных рядов

| Занятие | Тема | Материалы |
|---------|------|-----------|
| Лекция 5 - 1/2 | Методы машинного обучения для предсказания ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/05-1%20%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D0%BC%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B4%D0%BB%D1%8F%20%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%D0%B0%20%D0%92%D0%A0.pptx) |
| Лекция 5 - 2/2 | Оценка надежности предсказаний ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/05-2%20%D0%9E%D1%86%D0%B5%D0%BD%D0%BA%D0%B0%20%D0%BD%D0%B0%D0%B4%D0%B5%D0%B6%D0%BD%D0%BE%D1%81%D1%82%D0%B8%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B9%20%D0%92%D0%A0.pptx) |
| Практика 5 | ВР Оценка надежности предсказаний ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/05%20-%20Advanced%20in%20Forecast.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/05%20-%20Advanced%20in%20Forecast.ipynb) |
| Практика 6 | ML предсказание | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/06%20-%20ML%20forecast.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/06%20-%20ML%20forecast.ipynb) |
| Лекция 6 - 1/2 | Многомерные ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/README.md) |
| Лекция 6 - 2/2 | Обнаружение аномалий во ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/06-2%20%D0%9E%D0%B1%D0%BD%D0%B0%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B0%D0%BD%D0%BE%D0%BC%D0%B0%D0%BB%D0%B8%D0%B9.pptx) |
| Практика 7 | Обнаружение аномалий и другие задачи АВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/07%20anomaly_detect_TS.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/07%20anomaly_detect_TS.ipynb) |

### Модуль 4 - Использование искусственных нейронных сетей глубокого обучения в задачах анализа временных рядов

| Занятие | Тема | Материалы |
|---------|------|-----------|
| Лекция 7 - 1/2 | Полносвязные архитектуры нейронных сетей | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/07-1%20MLP.pptx) |
| Лекция 7 - 2/2 | Рекуррентные архитектуры нейронных сетей | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/07-2%20RNN.pptx) |
| Лекция 8 | Архитектуры - трансфомеры | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/07-3%20Transformers.pptx) |
| Практика 8 | Предсказания с использованием нейронных сетей | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/08%20-%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%92%D0%A0%20%D1%81%20DL.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/08%20-%20%D0%BF%D1%80%D0%B5%D0%B4%D1%81%D0%BA%D0%B0%D0%B7%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D0%B9%20%D0%92%D0%A0%20%D1%81%20DL.ipynb) |

### Дополнительные темы

| Занятие | Тема | Материалы |
|---------|------|-----------|
| Практика | Моделирование ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/00%20%D0%9C%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/00%20%D0%9C%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85%20%D1%80%D1%8F%D0%B4%D0%BE%D0%B2.ipynb) |
| Лекция | Классификация ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/09%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20%D0%92%D0%A0.pptx) |
| Практика | Data Driven Классификация ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/09%20%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B2%20SKTime.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/09%20%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20%D0%B2%20SKTime.ipynb) |
| Практика | Deep Learning Классификация ВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/09%20-%202%20TSAI_%D0%92%D0%92%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5.ipynb) / [Colab](https://colab.research.google.com/github/MVRonkin/TimeSeriesCourse/blob/main/2026/WS/09%20-%202%20TSAI_%D0%92%D0%92%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5.ipynb) |
| Лекция | Непараметрические методы АВР | [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/LEC/10%20%D0%9D%D0%B5%D0%BF%D0%B0%D1%80%D0%B0%D0%BC%D0%B5%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D1%81%20%D0%92%D0%A0.pptx) |

## ИТОГОВЫЙ КОНТРОЛЬ

**Основной репозиторий курса:** [GitHub](https://github.com/MVRonkin/TimeSeriesCourse/blob/main/2026/README.md)


# Литература
## Основная литература:
* [Rob J H Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/) - основной учебник
* [Rob J H and G Athanasopoulos Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/) - расширенное описание глав, посвященных класическому АВР
* [Nixtla statsforecas](https://nixtlaverse.nixtla.io/statsforecast/index.html) и  [GitHub документация](https://github.com/Nixtla/statsforecast/tree/main/nbs/docs) - основной фреймворк по классике, [GitHub docs2](https://github.com/Nixtla/statsforecast/tree/main/docs), [GitHub Experiments](https://github.com/Nixtla/statsforecast/tree/main/experiments)
* [Nixtla mlforecast](https://nixtlaverse.nixtla.io/mlforecast/index.html) и [GitHub docs](https://github.com/Nixtla/mlforecast/tree/main/nbs/docs) и [GitHub docs2](https://github.com/Nixtla/mlforecast/tree/main/docs) 
* [Nixtla neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast) и [GitHub docs](https://github.com/Nixtla/neuralforecast/tree/main/nbs/docs), [GitHub docs2](https://github.com/Nixtla/neuralforecast/tree/main/docs) and [GitHub Experiments](https://github.com/Nixtla/neuralforecast/tree/main/experiments)
* [Nixtla blog github](https://github.com/Nixtla/nixtla_blog_examples/tree/main) и [тут](https://github.com/Nixtla/transfer-learning-time-series/tree/main/nbs)
* [Blog ODS TSA](https://habr.com/ru/companies/ods/articles/327242/), [GitHub Ru](https://github.com/Yorko/mlcourse.ai/tree/main/jupyter_russian/topic09_time_series), [GitHub En](https://github.com/Yorko/mlcourse.ai/tree/main/jupyter_english/topic09_time_series)
* [ШАД Ml handbook Chs 10.2-10.5](https://education.yandex.ru/handbook/ml/article/vremennye-ryady)
* [Machine Learning for Time Series (Master MVA)](https://www.laurentoudre.fr/ast.html)

## Также рекомендуемая литература 
* [Блог введение в анализ временных рядов](https://www.dmitrymakarov.ru/intro/time-series/)
* [Bianchi F M Time series analysis with Python](https://filippomb.github.io/python-time-series-handbook/notebooks/00/intro.html#) 
[GitHub](https://github.com/FilippoMB/python-time-series-handbook /) -  простое Pythonic Way  изложение классического моделирования ВР
* [skforecast](https://skforecast.org/0.19.1/examples/examples_english.html) и [примеры к skforecast](/ https://cienciadedatos.net/documentos/py27-time-series-forecasting-python-scikitlearn.html) - полезное и хорошее описание примеров предсказания ВР
* [statsmodels.tsa](https://www.statsmodels.org/stable/tsa.html) и [user guide](/ https://www.statsmodels.org/stable/user-guide.html#time-series-analysis) - классический фреймворк моделирования ВР
* [прекрасные туториалы skforecast](https://skforecast.org/0.19.1/examples/examples_english.html) / [GitHub](https://github.com/skforecast/skforecast/tree/master/docs) / [Mirror](https://cienciadedatos.net/en/forecasting-python)
* [Простая статья по TSA](https://ivan-shamaev.ru/time-series-analysis-forecasting-and-models-python-libraries/)

##  Дополнительная литература 
* [Forecasting and Analytics with the Augmented Dynamic Adaptive Model (ADAM)](https://openforecast.org/adam/)
* https://clauswilke.com/dataviz/ 
* https://github.com/M-3LAB/awesome-industrial-anomaly-detection
* https://nicolarighetti.github.io/Time-Series-Analysis-With-R/
* https://rc2e.com/timeseriesanalysis
* https://mlcourse.ai/book/topic09/topic9_part1_time_series_python.html
* https://wesmckinney.com/book/time-series
* https://jakevdp.github.io/PythoмnDataScienceHandbook/03.11-working-with-time-series.html
* Time Series with Deep Learning Quick https://dl.leima.is/time-series/
* https://github.com/aeon-toolkit/aeon-tutorials/tree/main
* https://github.com/sktime/python_brasil_2025
* https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/
* http://www.stat.ucla.edu/~frederic/415/S23/tsa4.pdf
* https://www.kaggle.com/code/konradb/ts-0-the-basics
* https://github.com/ajitsingh98/Time-Series-Analysis-and-Forecasting-with-Python
* https://github.com/youssefHosni/Hands-On-Time-Series-Analysis-with-Python
* https://www.causalmlbook.com/time-series-forecasting.html
* https://github.com/USTCAGI/Awesome-Papers-Time-Series-Forecasting
* https://chaos.phys.msu.ru/loskutov/PDF/Lectures_time_series_analysis.pdf
* https://lib.ulstu.ru/venec/disk/2022/6.pdf
* https://github.com/thuml/Large-Time-Series-Model
* https://github.com/thuml/Time-Series-Library
* https://github.com/SalesforceAIResearch/uni2ts
* https://github.com/marcopeix/TimeSeriesForecastingUsingFoundationModels

## Какие то курсы
* https://github.com/LinkedInLearning/python-for-time-series-forecasting-5246009/tree/main
* https://github.com/aromanenko/ATSF/tree/main
* https://github.com/gheisenberg/TSF
* https://github.com/datons/TS/tree/
* https://github.com/oscar-defelice/TimeSeries-lectures
* https://github.com/j-adamczyk/ml_time_series_forecasting_course/
* https://github.com/trainindata/feature-engineering-for-time-series-forecasting/

## Инструменты универсальные и предсказания
* https://www.sktime.net/en/stable/
* https://skforecast.org/
* https://nixtlaverse.nixtla.io/statsforecast/index.html /https://github.com/Nixtla/statsforecast
* https://www.statsmodels.org/stable/tsa.html / https://www.statsmodels.org/stable/user-guide.html#time-series-analysis
* https://unit8co.github.io/darts/
* https://lavinei.github.io/pybats/
* https://github.com/Y-Research-SBU/TimeSeriesScientist/tree/main

* [25 years of open source forecasting software 2025](https://robjhyndman.com/seminars/osfs25.html)

## Инструменты TSC
* https://aeon-toolkit.org/
* https://timeseriesai.github.io/tsai/
* https://tsfel.readthedocs.io/
* https://github.com/predict-idlab/tsflex / https://predict-idlab.github.io/tsflex/
* https://github.com/valeman/awesome-conformal-prediction
* https://github.com/thuml/Time-Series-Library
* https://github.com/ritvikmath/Time-Series-Analysis
* 


## Инструменты обнаружения аномалий, перегиба, дрейфа новизны и тд
* https://riverml.xyz / https://habr.com/ru/companies/glowbyte/articles/681772/
* https://github.com/agpenas/tstrends
* https://tom-doerr.github.io/repo_posts/2025/04/26/Nixtla-statsforecast.html

* https://github.com/thekimk/All-About-Time-Series-Analysis/

## Тутроиалы
* https://github.com/aeon-toolkit/aeon-tutorials/tree/main
* https://github.com/sktime/python_brasil_2025 
* https://github.com/sktime/sktime-workshop-pycon-colombia-2025/tree/main/notebooks
* https://github.com/Nixtla/blog/
* https://github.com/Nixtla/fpp3-python
* https://www.pymc-labs.com/blog-posts/probabilistic-forecasting
* https://www.pymc.io/projects/docs/en/latest/api/distributions/timeseries.html
* https://juanitorduz.github.io/short_time_series_pymc/
* https://neuralprophet.com/tutorials/index.html
* https://skforecast.org/0.19.1/examples/examples_english.html
* https://readmedium.com/bayesian-time-series-forecasting-in-python-with-the-ubers-orbit-package-1d3b7ff482dd
* https://github.com/aangelopoulos/conformal-prediction/tree/main
* https://towardsdatascience.com/uncertainty-quantification-in-time-series-forecasting-c9599d15b08b/
* https://readmedium.com/time-series-forecasting-with-conformal-prediction-intervals-scikit-learn-is-all-you-need-4b68143a027a
* https://github.com/cerlymarco/MEDIUM_NoteBook
* https://github.com/cerlymarco/tspiral
* https://juanitorduz.github.io/tsb_numpyro/ https://juanitorduz.github.io/croston_numpyro/
* https://github.com/valeman/Transformers_And_LLM_Are_What_You_Dont_Need
* https://github.com/cuge1995/awesome-time-series
