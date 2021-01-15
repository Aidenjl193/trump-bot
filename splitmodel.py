CHUNK_SIZE = 104857600
file_number = 1
with open('gpt2_medium_tweetr_4.pt', 'rb') as f:
    chunk = f.read(CHUNK_SIZE)
    while chunk:
        with open(f'model_part_{str(file_number)}.bin', "wb") as chunk_file:
            chunk_file.write(chunk)
        file_number += 1
        chunk = f.read(CHUNK_SIZE)