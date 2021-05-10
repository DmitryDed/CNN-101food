## Лабораторная работа #5.
##        Решение задачи классификации изображений из набора данных Food-101 использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning


## 1. С использованием примера, техники обучения Transfer Learning, оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить  нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101

```python
tf
```

## Графики
![image](https://user-images.githubusercontent.com/81873177/117677924-93b24f00-b1b7-11eb-808b-deb112bf4ee1.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracy50.svg)

График функции потерь

![SVG example](./grafs/epoch_loss50.svg)

## 2. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 1.

## Графики
![image](https://user-images.githubusercontent.com/81873177/117677701-5ea5fc80-b1b7-11eb-9cb7-60aef9a662ce.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracy10.svg)

График функции потерь

![SVG example](./grafs/epoch_loss10.svg)

## Анализ результатов:
