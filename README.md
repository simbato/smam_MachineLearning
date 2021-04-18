# Klasifikacija glazbe po žanru

Modeli su podijeljeni u dvije bilježnice:
* `klasicni_modeli.ipynb`: klasični modeli strojnog učenja
* `cnn_modeli.ipynb`: konvolucijske neuronske mreže

Bilježnice je najlakše pokrenuti na Kaggle-u, pritom koristeći unaprijed generirane datasetove:
* [FMA_small](https://www.kaggle.com/aaronyim/fma-small)
* [handcrafted](https://www.kaggle.com/dataset/206d79191cb67e3d506989a8d420f83172028279e0546d789829ed041eb16c7c)
* [fma-small-s-db](https://www.kaggle.com/dataset/8885d97c06ee6c451ac372fd215c2d004010bdc32770886d5c33a46e49717a64)

Da biste samostalno generirali dataset, potrebno je:

1. Skinuti pjesme i metapodatke s Github repozitorija [FMA](https://github.com/mdeff/fma) dataseta:
   * `fma_metadata.zip`
   * `fma_small.zip`
1. Raspakirati `fma_small.zip` unutar repozitorija
1. Instalirati FFmpeg:
   * Ubuntu: `sudo apt install ffmpeg`
1. Podesiti Python virtualno okruženje i instalirati potrebne pakete:
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
3. Pokrenuti `python generate.py`
4. Pokrenuti `python generate_handcrafted.py`