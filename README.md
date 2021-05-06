## Лабораторная работа #5.
##        Решение задачи классификации изображений из набора данных Food-101 использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning


## 1. С использованием примера [2], техники обучения Transfer Learning [1], оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить  нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101

```python
tf.keras.layers.experimental.preprocessing.Random
```

## Графики
![image](https://user-images.githubusercontent.com/81873177/117008405-5a7b6a00-acf3-11eb-8806-22be2b80788b.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracyhv.svg)
![image](https://user-images.githubusercontent.com/81873177/117008376-53ecf280-acf3-11eb-8c75-6d08cbe69810.png)

График функции потерь

![SVG example](./grafs/epoch_losshv.svg)
![image](https://user-images.githubusercontent.com/81873177/117008364-4f283e80-acf3-11eb-8b90-74b0351efe3c.png)
## 2. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 1.


## Анализ результатов:
