import os
import pandas as pd

def check_posters():
    """Cek poster mana yang sudah tersedia"""
    
    # Load movies
    movies_df = pd.read_csv('data/movies.csv')
    movie_ids = movies_df['movieId'].head(1000).tolist()
    
    poster_folder = 'static/posters'
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    found = []
    missing = []
    
    for movie_id in movie_ids:
        poster_exists = False
        for ext in extensions:
            if os.path.exists(os.path.join(poster_folder, f"{movie_id}{ext}")):
                poster_exists = True
                found.append(movie_id)
                break
        
        if not poster_exists:
            missing.append(movie_id)
    
    print(f"✅ Poster ditemukan: {len(found)} film")
    print(f"❌ Poster tidak ada: {len(missing)} film")
    print(f"\nMovie ID tanpa poster (10 pertama): {missing[:10]}")
    
    # Simpan daftar yang hilang
    with open('missing_posters.txt', 'w') as f:
        for mid in missing:
            title = movies_df[movies_df['movieId'] == mid]['title'].values[0]
            f.write(f"{mid}: {title}\n")
    
    print(f"\nDaftar lengkap disimpan di 'missing_posters.txt'")

if __name__ == '__main__':
    # Buat folder posters jika belum ada
    os.makedirs('static/posters', exist_ok=True)
    check_posters()
