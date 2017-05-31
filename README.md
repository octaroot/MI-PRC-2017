# MI-PRC 2016/17
## Cannyho hranový detektor

Realizace na paltformě CUDA s velkou řadou optimalizací (masivní many-core paralelismus).


Překlad pomocí

```
nvcc main.cu --gpu-architecture=sm_50 --use_fast_math --optimize 3
```

Poznámka: `--gpu-architecture=sm_50` změnit dle možností cílové GPU karty.