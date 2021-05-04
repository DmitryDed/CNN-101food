## Лабораторная работа #4.
##         Использование техник аугментации данных для улучшения сходимости процесса обучения нейронной сети на примере решения задачи классификации Food-101


## 1. С использованием, техники обучения Transfer Learning и оптимальной политики изменения темпа обучения, определенной в ходе выполнения лабораторной #3, обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием следующих техник аугментации данных:
## a. Случайное горизонтальное и вертикальное отображение 

```python
tf.keras.layers.experimental.preprocessing.RandomFlip(mode="vertical", seed=None, name=None)
tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=None, name=None)
tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal_and_vertical", seed=None, name=None)
```

## Графики
![image](https://user-images.githubusercontent.com/81873177/117008405-5a7b6a00-acf3-11eb-8806-22be2b80788b.png)

График точности

![SVG example](./grafs/epoch_categorical_accuracyhv.svg)
![image](https://user-images.githubusercontent.com/81873177/117008376-53ecf280-acf3-11eb-8c75-6d08cbe69810.png)

График функции потерь

![SVG example](./grafs/epoch_losshv.svg)
![image](https://user-images.githubusercontent.com/81873177/117008364-4f283e80-acf3-11eb-8b90-74b0351efe3c.png)

## b. Использование случайной части изображения 

## Графики
![image](https://user-images.githubusercontent.com/81873177/117046951-f91ac180-ad19-11eb-9310-1a213dae64a1.png)



График точности
![SVG example](./grafs/epoch_categorical_accuracyCR.svg)
![image](https://user-images.githubusercontent.com/81873177/117046917-ed2eff80-ad19-11eb-954d-a75333031661.png)


График функции потерь
![SVG example](./grafs/epoch_lossCR.svg)
![image](https://user-images.githubusercontent.com/81873177/117046879-e43e2e00-ad19-11eb-95a1-ebf1da727e45.png)


## c. Поворот на случайный угол 
 R1, R2, R3 соответственно:
```python
tf.keras.layers.experimental.preprocessing.RandomRotation(0.25, fill_mode='reflect', interpolation='bilinear', seed=None, name=None, fill_value=0.0)
tf.keras.layers.experimental.preprocessing.RandomRotation(0.12, fill_mode='reflect', interpolation='bilinear', seed=None, name=None, fill_value=0.0)
tf.keras.layers.experimental.preprocessing.RandomRotation(0.03, fill_mode='reflect', interpolation='bilinear', seed=None, name=None, fill_value=0.0)
```

## Графики
![image](https://user-images.githubusercontent.com/81873177/117006002-8f39f200-acf0-11eb-8a05-89075ea37611.png)


График точности

![SVG example](./grafs/epoch_categorical_accuracyR.svg)

![image](https://user-images.githubusercontent.com/81873177/117007038-d4125880-acf1-11eb-98ff-6795a92fcb7d.png)


График функции потерь

![SVG example](./grafs/epoch_lossR.svg)

![image](https://user-images.githubusercontent.com/81873177/117007074-dffe1a80-acf1-11eb-926e-f27c5cfe1ab1.png)


## 2. Для каждой индивидуальной техники аугментации определить оптимальный набор параметров


## 3. Обучить нейронную сеть с использованием оптимальных техник аугментации данных 2a-с совместно

## Анализ результатов:

