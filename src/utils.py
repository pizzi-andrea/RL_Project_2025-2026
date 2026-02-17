import cv2 as cv
import os
from pathlib import Path

def create_video_from_frames(input_folder, output_path, fps=30):
    """
    Legge i frame da una cartella e salva un video.
    
    Args:
        input_folder (str): Path della cartella contenente i frame (es. 'path/to/frames').
        output_path (str): Path completo di destinazione (es. 'output/video.avi').
        fps (int): Frame per secondo.
    """
    images = sorted([Path(img) for img in os.listdir(input_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    images.sort(key=lambda x: x.name)
    if not images:
        print("Nessuna immagine trovata nella cartella.")
        return
    input_folder = Path(input_folder)
    # Leggi la prima immagine per determinare le dimensioni
    frame = cv.imread( input_folder.joinpath(str(images[0])))
    height, width, layers = frame.shape

    # Configurazione codec (XVID è standard per .avi, come usato nel tuo progetto)
    #
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creazione video in corso: {len(images)} frame...")
    
    for image in images:
        img_path = input_folder.joinpath(str(image))
        video_writer.write(cv.imread(str(img_path)))

    video_writer.release()
    print(f"Video salvato con successo in: {str(output_path)}")