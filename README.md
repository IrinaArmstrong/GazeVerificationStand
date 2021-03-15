# GazeVerificationStand

Тестовый вариант рабочего стенда для верификационной части Инструментального комплекса анализа движений глаз для задач интерактивной рефлекторной идентификации (ИРИ).

## Оглавление

0. [Общее описание](#Общее-описание)
1. [Состав модулей](#Состав-модулей)
    1. [Конфигурация и пути](#Конфигурация)
    2. [Запуск](#Запуск)
    3. [Обработка данных](#Обработка)
    4. [Классификация движений глаз](#Классификация-движений)
    5. [Верификация](#Верификация)
    6. [Jupiter notebooks](#Jupiter-notebooks)
2. [Визуализация](#Визуализация)
3. [Режимы работы](#Режимы-работы)
4. [Файловая структура](#Файловая-структура)
5. [Дополнения](#Дополнения)
  ____
  
## Общее описание

Стенд состоит из пяти логически и программно разделенных модулей. Каждый из них отвечает за отдельный реализованный алгоритм или обеспечивает корректное функционирование всей системы в целом.
* `Модуль конфигурации` - отвечает за конфигурационный файл, который используется для хранения настроек стенда.
* `Модуль запуска` - входная точка для запуска работы стенда с выбранной конфигурацией.
* `Модуль обработки данных` - реализует все необходимые утилиты и методы для работы с "сырыми" данными с трекера.
* `Модуль классификации движений глаз` - содержит алгоритм и сопутствующие ему функции для классификации временных рядов взглядов в последовательность определенных макродвижений человеческого глаза.
* `Модуль верификации личности` - отвечает непосредственно за реализацию самой процедуры идентификации человека как "владельца устройства".
* `Дополнительный модель с интерактивными Jupiter ноутбуками` - полезен для демонстрации промежуточных результатов пайплайна алгоритмов и оценки качества.

## Состав модулей

### Модуль конфигурации


### Модуль запуска


### Модуль обработки данных


### Модуль классификации движений глаз


### Модуль верификации личности


### Дополнительный модель с интерактивными Jupiter ноутбуками


## Визуализация


## Режимы работы


## Файловая структура