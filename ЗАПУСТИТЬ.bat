@echo off
chcp 65001 > nul
title Установка и запуск ML проекта
color 0A

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║    ОИИ — Лабораторные работы №9-14       ║
echo  ║    Автоматическая установка              ║
echo  ╚══════════════════════════════════════════╝
echo.

echo  [1/3] Проверяем Python...
python --version
if errorlevel 1 (
    echo.
    echo  ОШИБКА: Python не установлен!
    echo  Скачай с https://www.python.org/downloads/
    echo  При установке ОБЯЗАТЕЛЬНО поставь галочку "Add to PATH"
    pause
    exit /b 1
)

echo.
echo  [2/3] Устанавливаем библиотеки...
pip install -r requirements.txt -q
echo  Готово!

echo.
echo  [3/3] Открываем сайт-презентацию...
start "" python -m http.server 8080 --directory site
timeout /t 2 /nobreak > nul
start "" http://localhost:8080

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║  Сайт открылся в браузере!               ║
echo  ║  Адрес: http://localhost:8080            ║
echo  ║                                          ║
echo  ║  Чтобы запустить задание, в терминале:   ║
echo  ║  python week12_kmeans_clustering.py      ║
echo  ║  python week13_svm_classification.py     ║
echo  ║  python week9_neural_network.py          ║
echo  ║  python week11_knn_classifier.py         ║
echo  ║  python week14_reinforcement_learning.py ║
echo  ╚══════════════════════════════════════════╝
echo.
echo  Не закрывай это окно пока смотришь сайт!
pause
