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
![image](https://user-images.githubusercontent.com/81873177/117062390-0ccf2380-ad2c-11eb-9502-42613296002c.png)

Изображение, полученное при использовании оптимальных параметров:

![flip-28](https://user-images.githubusercontent.com/81873177/117055295-aba35200-ad23-11eb-8c32-e3825e0e348e.jpg)


## b. Использование случайной части изображения 

## Графики
![image](https://user-images.githubusercontent.com/81873177/117046951-f91ac180-ad19-11eb-9310-1a213dae64a1.png)


График точности
![SVG example](./grafs/epoch_categorical_accuracyCR.svg)
![image](https://user-images.githubusercontent.com/81873177/117046917-ed2eff80-ad19-11eb-954d-a75333031661.png)


График функции потерь
![SVG example](./grafs/epoch_lossCR.svg)
![image](https://user-images.githubusercontent.com/81873177/117046879-e43e2e00-ad19-11eb-95a1-ebf1da727e45.png)

Изображение, полученное при использовании оптимальных параметров:

![crop-28](https://user-images.githubusercontent.com/81873177/117055360-beb62200-ad23-11eb-939b-cda28f066862.jpg)


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

Изображение, полученное при использовании оптимальных параметров:

![3](https://user-images.githubusercontent.com/81873177/117055414-cbd31100-ad23-11eb-8272-5b964eed94d8.jpg)


## 2. Для каждой индивидуальной техники аугментации определить оптимальный набор параметров

```python
tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal", seed=None, name=None)
tf.keras.layers.experimental.preprocessing.RandomRotation(0.03, fill_mode='reflect', interpolation='bilinear', seed=None, name=None, fill_value=0.0)
tf.keras.layers.experimental.preprocessing.RandomCrop(CROPSIZE_HEIGHT, CROPSIZE_WIDTH)
```
## 3. Обучить нейронную сеть с использованием оптимальных техник аугментации данных 2a-с совместно
## Графики
![image](https://user-images.githubusercontent.com/81873177/117060320-7437a400-ad29-11eb-98ab-f425f1d6db00.png)


График точности

![SVG example](./grafs/epoch_categorical_accuracym.svg)

![image](https://user-images.githubusercontent.com/81873177/117060447-9cbf9e00-ad29-11eb-94cb-0d747af97dcf.png)


График функции потерь

![SVG example](./grafs/epoch_lossm.svg)

![image](https://user-images.githubusercontent.com/81873177/117060478-a6490600-ad29-11eb-96c1-41e81fdef46d.png)

Изображение, полученное при использовании оптимальных параметров:

![4](https://user-images.githubusercontent.com/81873177/117055445-d4c3e280-ad23-11eb-9e41-137099ab8fa6.jpg)

## Анализ результатов:

1. Исследовав график точности и график функции потерь, можно прийти к выводу, что параметр 'horizontal' является оптимальным, так как имеет наивысшие значения на графике точности (67.66%).
2. Используя технику аугментации данных RandomCrop, можно сделать вывод, что наиболее оптимальными размерами изображения оказались 274х274, так как на графике точности принимает наивысшие значения: 65.31%. Также методом "проб и ошибок" выяснилось, что если брать значения намного большие или намного меньшие, чем наши, то мы потеряем качество изображения.
3. Используя технику аугментации RandomRotation, выяснилось, что наиболее оптимальными параметрами являются: (0.03, fill_mode='reflect', interpolation='bilinear', seed=None, name=None, fill_value=0.0). Значения на графике точности при этом были 67.12%.
4. При комбинации техник аугментации достигнута точность 65.55%. Сравнивая с RandomFlip, точность уменьшилась на 2,11%, с RandomRotation -  уменьшилась на 1.57%, с RandomCrop - увеличилась на 0,24%.

