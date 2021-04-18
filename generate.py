import os
import numpy as np
import librosa
import warnings
from pathlib import Path
warnings.simplefilter('ignore', UserWarning)

# lokacija raspakiranog fma_small direktorija
fma_dir = 'fma_small'

def get_filenames(dir, extension=''):
    result = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(extension):
                result.append((filename, os.path.join(root, filename)))
    return result

if __name__ == "__main__":
    for dir in ['S_DB', 'raw']:
        Path(dir).mkdir(parents=True, exist_ok=True)

    files = get_filenames(fma_dir, '.mp3')
    for i in range(len(files)):
        track_id = int(files[i][0][:-4])
        raw_file = f'raw/{track_id}_y_sr.npy'
        s_db_file = f'S_DB/{track_id}.npy'
        if not os.path.isfile(raw_file) or not os.path.isfile(s_db_file):
            try:
                y, sr = librosa.load(files[i][1])
                S = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128)
                S_DB = librosa.power_to_db(S, ref=np.max)
                np.save(raw_file, (y, sr))
                np.save(s_db_file, S_DB)
                print(f'Saved {track_id}')
            except Exception as e:
                print(e)
                with open('failed.txt', 'a') as f:
                    f.write(f'{track_id}\n')
                print(f'Failed {track_id}')
        print(f'{(i+1)/len(files)*100:.1f}% done')



