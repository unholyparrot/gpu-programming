Реализован фильтр "Автоконтраст"
    1. Рассчитывается яркость пикселей по формуле Y = 0.2125 * R + 0.7154 * G + 0.0721 * B
    2. Строится гистограмма яркости изображения
    3. На основе гистограммы яркостей вычисляются параметры растяжения яркости всего изображения 
        (примерно как тут https://en.wikipedia.org/wiki/Histogram_equalization)
    4. Полученные параметры растяжения применяются к исходному изображению.

    В изначальной формулировке неясно до конца, к чему нужно применять параметры растяжения: 
    к исходному RGB изображению, или только к яркости. В итоговой реализации в результате 
    получается RGB изображение, каждый канал которого растянут одинаковым образом 
    (с параметрами с шага 3).

Детали проекта
    * Задание выполнено на CUDA (GPU часть)
    * Для чтения/записи изображений используется stb_image
    * Сборка с помощью CMake:

        mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release ..
        cmake --build .

Запуск приложения
    * На вход подается путь к изображению `path_to_image`
    * На выходе результаты записываются в 2 файла `out_cpu.png` и `out_gpu.png`
    * Для запуска замеров времени между разным количеством OMP потоков и
    реализациями гистограмм используется ключ `-b`
    * Help: `-h`
    
        ./main path_to_image [-b]
    
Спецификация ПК
    * i5-8265U @ 1.60GHz, 4 Cores 8 Threads 
    * 8GB RAM
    * NVIDIA GTX 1050-Ti (Mobile Max-Q) 2GB VRAM
    * Сборка и работоспособность тестировались на Windows 11

Замеры времени на изображении размером 1459x2048 пикселей
    * CPU:
        Elapsed time on CPU, 1 OMP threads:     25935   [microseconds]
        Elapsed time on CPU, 4 OMP threads:     14386   [microseconds]
        Elapsed time on CPU, 8 OMP threads:     11108   [microseconds]

    * GPU, mode 0 -- one histogram for all thread blocks in global memory:
        Elapsed time on GPU, mode 0, total process:     635881  [microseconds]
        Elapsed time on GPU, mode 0, without memcpy:    3796    [microseconds]
        Elapsed time on GPU, mode 0, only memcpy:       632085  [microseconds]
        Elapsed time on GPU, mode 0, histogram calc:    1743    [microseconds]

    * GPU, mode 1 -- local histograms for each thread block in global memory:
        Elapsed time on GPU, mode 1, total process:     10144   [microseconds]
        Elapsed time on GPU, mode 1, without memcpy:    3181    [microseconds]
        Elapsed time on GPU, mode 1, only memcpy:       6963    [microseconds]
        Elapsed time on GPU, mode 1, histogram calc:    1219    [microseconds]

    * GPU, mode 2 -- local histograms for each thread block in shared memory:
        Elapsed time on GPU, mode 2, total process:     10078   [microseconds]
        Elapsed time on GPU, mode 2, without memcpy:    2989    [microseconds]
        Elapsed time on GPU, mode 2, only memcpy:       7089    [microseconds]
        Elapsed time on GPU, mode 2, histogram calc:    1031    [microseconds]
    
    * Итоговое время работы:
        CPU, 1 поток            --- 25935 μs
        CPU, 4 потока           --- 14386 μs
        GPU, лучшая реализация  --- 10078 μs
    
Оптимизации
    Оптимизировалась функция подсчета гистограммы яркости изображения.
    Все варианты реализаций представлены в `histogram.cu`
    
    1. Гистограмма хранится в глобальной памяти, все нити работают с ней напрямую через atomicAdd.
    Реализация: `void histogram_global_GPU`

    Время работы этой реализации: 1.7 ms на GPU на изображении 1459x2048

    2. Для каждого блока нитей выделяется свой участок в глобальной памяти.
    Нити из одного блока работают с этим участком напрямую через atomicAdd.
    Далее, отдельным ядром на 256 нитей все локальные гистограммы суммируются в итоговую.
    Реализация: `void histogram_local_globalmem_GPU`--- локальные гистограммы
    `void histogram_finalize_GPU` --- суммирование гистограмм

    Время работы этой реализации: 1.2 ms на GPU на изображении 1459x2048

    3. Каждый блок нитей работает в shared памяти, увеличивая через atomicAdd. Затем,
    каждая полученная локальная гистограмма копируется в свой участок в глобальной памяти.
    Далее, как в (2), отдельным ядром на 256 нитей все локальные гистограммы суммируются в итоговую.
    Реализация: `void histogram_local_sharedmem_GPU`--- локальные гистограммы
    `void histogram_finalize_GPU` --- суммирование гистограмм

    Время работы этой реализации: 1.0 ms на GPU на изображении 1459x2048

    Ускорение 3й версии по сравнению со 2й достигается засчет аппаратной поддержки 
    архитектуры Maxwell на моей видеокарте.

    Итоговое ускорение по сравнению с наивной реализацией --- в ~1.7 раз.

Для оптимизации использовалась статья https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
