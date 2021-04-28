## Лабораторная работа #3.
##          Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Food-101 с использованием техники обучения Transfer Learning
## 1. С использованием и техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием фиксированных темпов обучения 0.01, 0.001, 0.0001.
 

## Графики
![image](https://user-images.githubusercontent.com/81873177/116250303-46c18800-a776-11eb-8407-f5c1215895b5.png)

График точности
![SVG example](./grafs/epoch_categorical_accuracy2.svg)
![image](https://user-images.githubusercontent.com/81873177/116288822-91ed9200-a79a-11eb-9859-1cc42d288392.png)


График функции потерь
![SVG example](./grafs/epoch_loss2.svg)
![image](https://user-images.githubusercontent.com/81873177/116288857-9914a000-a79a-11eb-8258-6bc2353e2e62.png)


## 2. Реализовать и применить в обучении следующие политики изменения темпа обучения, а также определить оптимальные параметры для каждой политики:
## a. Косинусное затухание (Cosine Decay) 

```python
tf.keras.experimental.CosineDecay(initial_learning_rate, decay_steps, alpha=0.0)
```
initial_learning_rate	- начальная скорость обучения

decay_steps	- количество шагов

alpha -	минимальное значение скорости обучения

## Графики
![image](https://user-images.githubusercontent.com/81873177/116254807-55aa3980-a77a-11eb-8246-d9b70013396e.png)

График точности
![SVG example](./grafs/epoch_categorical_accuracy3a.svg)
![image](https://user-images.githubusercontent.com/81873177/116276753-7a5bdc80-a78d-11eb-92fb-3da5bb33ee50.png)


График функции потерь

![SVG example](./grafs/epoch_loss3a.svg)

![image](https://user-images.githubusercontent.com/81873177/116276771-80ea5400-a78d-11eb-9651-aa989a53fa88.png)

Графики темпов обучения 

![SVG example](./grafs/cd3b1.svg)
![image](https://user-images.githubusercontent.com/81873177/116308489-37f7c700-a7b0-11eb-9f4c-e848d6988167.png)

## b. Косинусное затухание с перезапусками (Cosine Decay with Restarts) 
```python
tf.keras.experimental.CosineDecayRestarts(initial_learning_rate, first_decay_steps, t_mul=2.0, m_mul=1.0)
```
initial_learning_rate	- начальная скорость обучения

decay_steps	- количество шагов

t_mul	- используется для определения количества итераций в i-м периоде

m_mul	- используется для получения начальной скорости обучения i-го периода

## Графики
![image](https://user-images.githubusercontent.com/81873177/116306223-7770e400-a7ad-11eb-9dda-966500838c24.png)


График точности
![SVG example](./grafs/epoch_categorical_accuracy3b.svg)

График функции потерь
![SVG example](./grafs/epoch_loss3b.svg)


Графики темпов обучения 
![SVG example](./grafs/epoch_learning_rate3b11.svg)
![SVG example](./grafs/3b11.svg)

## Анализ результатов:
1. Исследуя графики метрики точности и графики функции потерь, можно прийти к выводу, что в нашем случае шаг 0,0001 является оптимальным, так как на графике метрики точности наблюдаются наивысшие значения - 67.92%, а на графике функции потерь наименьшие - 1,197.

2а. Исследуя графики метрики точности и графики функции потерь для случая с косинусным затуханием (Cosine Decay),  можно сказать, что в нашем случае (initial_learning_rate=0.002, decay_steps=1000, alpha=0.03) шаг 0,002 является оптимальным, так как на графике метрики точности наблюдаются наивысшие значения - 67.99%, а на графике функции потерь наименьшие - 1,192.

2b. Исследуя графики метрики точности и графики функции потерь для случая с косинусным затуханием с перезапусками (Cosine Decay with Restarts), видим, что наивысшие значения метрики точности cosine decay with restarts достигаются примерно на 8 эпохе с параметрами  first_decay_steps = 1000, t_mul = 2.25, m_mul = 1.5 - 66%. Примерно на той же эпохе достигается наименьшее значение на графике функции потерь - 0,012. 
