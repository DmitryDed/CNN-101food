## Лабораторная работа #5.
##        Решение задачи классификации изображений из набора данных Food-101 использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning


## 1. С использованием примера, техники обучения Transfer Learning, оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить  нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101


## Графики
![image](https://user-images.githubusercontent.com/81873177/117677924-93b24f00-b1b7-11eb-808b-deb112bf4ee1.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracy50.svg)
![image](https://user-images.githubusercontent.com/81873177/117680067-8007e800-b1b9-11eb-855e-516a8c8e66c8.png)

График функции потерь

![SVG example](./grafs/epoch_loss50.svg)
![image](https://user-images.githubusercontent.com/81873177/117680082-85653280-b1b9-11eb-93e9-19546266af14.png)

## 2. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 1.

## Графики
![image](https://user-images.githubusercontent.com/81873177/117677701-5ea5fc80-b1b7-11eb-9cb7-60aef9a662ce.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracy10.svg)
![image](https://user-images.githubusercontent.com/81873177/117679977-69fa2780-b1b9-11eb-9744-1893611ef9ba.png)

График функции потерь

![SVG example](./grafs/epoch_loss10.svg)
![image](https://user-images.githubusercontent.com/81873177/117680004-6ff00880-b1b9-11eb-9242-eebadd1d01af.png)

## Анализ результатов:
При использовании техники обучения Fine Tuning с параметром темпа обучения, равным 1е-7, из графиков точности видно, что значения улучшились на 0,95% (augmentation with optimal settings - 68,10%, Fine Tuning - 69,05%)
