Репозиторий с проектом: https://github.com/dean1t/gpu-programming/tree/master/02

Реализован вариант задания с денойзингом

Детали проекта
    * Задание выполнено на CUDA (GPU часть)
    * Для чтения/записи изображений используется stb_image
        header-only библиотека https://github.com/nothings/stb
    * Сборка с помощью CMake:
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make

    * На Windows (чтобы собрался Release):
        mkdir build
        cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        cmake --build . --config Release
    
Запуск приложения
    * На вход подается путь к изображению `path_to_image.png`
    * На выходе результат записывается в `path_to_image_denoised.png`
    * Для корректных замеров времени используется ключ `-b`
    * Help: `-h`
    
        ./denoiser path_to_image [-b [X]]
    
Спецификация ПК
    * i5-8265U @ 1.60GHz, 4 Cores 8 Threads 
    * 8GB RAM
    * NVIDIA GTX 1050-Ti (Mobile Max-Q) 2GB VRAM
    * Сборка и работоспособность тестировались на Windows 11

Замеры времени на 100 запусках на изображении из MNIST размером 28x28 пикселей и описание оптимизаций
    1. Базовая версия (без оптимизаций)
    Elapsed time on one step:       10717   [microseconds]

    2. Перенос функций активации внутрь вычисления сверток
