from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import re
import string
import psutil
import pickle


from dotenv import load_dotenv
import os
import psycopg2

bert_model = None
bert_embeddings = None
food = None
def load_model():
    global bert_model, bert_embeddings, food, indices
    if bert_model is None or bert_embeddings is None or food is None:
        print("📦 Chargement de food_dataframe.pkl et bert_embeddings.pkl...")

        import urllib.request
        import pickle
        import numpy as np

        def download_if_missing(file_path, url):
            if not os.path.exists(file_path):
                print(f"⬇️ Téléchargement de {file_path}...")
                urllib.request.urlretrieve(url, file_path)
                print(f"✅ {file_path} téléchargé.")

        download_if_missing("bert_embeddings.pkl", "https://drive.google.com/uc?export=download&id=1yxIukVxwUuyuJj7bpp-ChPAEKd_tCTva")
        download_if_missing("food_dataframe.pkl", "https://drive.google.com/uc?export=download&id=1uR3OtKd4fHQMHjepBFdVA42ouCMJ0RuS")

        food = pd.read_pickle("food_dataframe.pkl")
        food['Name_clean'] = food['Name'].str.strip().str.lower()
        indices = pd.Series(food.index, index=food['Name_clean']).drop_duplicates()

# Chargement des embeddings compressés (déjà encodés)
        bert_embeddings = np.load("bert_embeddings.npz")["embeddings"]  # shape: (N, D), dtype: float16

        print_memory_usage("après chargement des embeddings (npz)")


       
def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"🧠 RAM utilisée {message}: {mem_mb:.2f} MB")


print("🔥 DÉMARRAGE API — version debug active")


# Charger .env
load_dotenv()

# Connexion PostgreSQL
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT'),
    sslmode='require'
)



app = Flask(__name__)
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Autorise toutes les origines


ratings = pd.read_sql('SELECT id_client AS "User_ID", id_plat AS "Food_ID", nb_etoile AS "Rating" FROM connected_restaurant.note_plat', conn)

# Chargement et préparation des données

# import urllib.request
# import pickle
# from sentence_transformers import SentenceTransformer, util

# def download_if_missing(file_path, url):
#     if not os.path.exists(file_path):
#         print(f"⬇️ Téléchargement de {file_path}...")
#         urllib.request.urlretrieve(url, file_path)
#         print(f"✅ {file_path} téléchargé.")

# download_if_missing("bert_embeddings.pkl", "https://drive.google.com/uc?export=download&id=1yxIukVxwUuyuJj7bpp-ChPAEKd_tCTva")
# download_if_missing("food_dataframe.pkl", "https://drive.google.com/uc?export=download&id=1uR3OtKd4fHQMHjepBFdVA42ouCMJ0RuS")

# bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# print("📦 Chargement de food_dataframe.pkl et bert_embeddings.pkl...")

# with open("bert_embeddings.pkl", "rb") as f:
#     bert_embeddings = pickle.load(f)

# food = pd.read_pickle("food_dataframe.pkl")
# food['Name_clean'] = food['Name'].str.strip().str.lower()
# indices = pd.Series(food.index, index=food['Name_clean']).drop_duplicates()


# Collaborative Filtering Setup

rating_matrix = ratings.pivot_table(index='User_ID', columns='Food_ID', values='Rating').fillna(0)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(csr_matrix(rating_matrix.values))
print_memory_usage("après entraînement KNN")

# --- Fonction pour afficher les utilisateurs similaires



# Mapping des noms vers indices pour content-based


@app.route('/hybrid_recommend', methods=['POST'])
def hybrid_recommend():
    load_model()
    print_memory_usage("pendant recommandation hybride")

    data = request.json
    user_id = int(data.get('user_id', -1))
    input_plats = data.get('plats', [])
    content_recs = []
    
    print("📩 Requête reçue sur /hybrid_recommend")
    print(f"📝 Plats reçus : {input_plats}")

    for plat in input_plats:
      plat_clean = plat.strip().lower()
      
      print(f"🔎 Recherche du plat : '{plat_clean}'")

      if plat_clean in indices:
          print(f"✅ Plat trouvé : '{plat_clean}'")
          idx = indices[plat_clean]
          if isinstance(idx, (pd.Series, np.ndarray, list)):
            idx = idx[0]

          plat_feature = food.iloc[idx]['C_Type'] + ' ' + food.iloc[idx]['Ingredient']
          plat_feature = plat_feature.strip().lower()
          query_embedding = bert_embeddings[idx].reshape(1, -1)  # float16
          row = cosine_similarity(query_embedding, bert_embeddings)[0]  # np.array de shape (400,)


          print(f"📏 Shape brute de cosine_sim[{idx}] = {getattr(row, 'shape', 'N/A')}")

        # 🔄 Flatten si nécessaire (3D, 2D...)
          while isinstance(row, np.ndarray) and row.ndim > 1:
            print(f"⚠️ Row avec ndim={row.ndim} → flatten : shape avant = {row.shape}")
            row = row[0]

          print(f"✅ Row final prête pour tri → shape = {getattr(row, 'shape', 'N/A')} | ndim = {getattr(row, 'ndim', 'N/A')}")

          sim_scores = list(enumerate(row))
          try:
            sim_scores = sorted(
                sim_scores,
                key=lambda x: x[1].item() if hasattr(x[1], "item")
                              else x[1][0] if isinstance(x[1], (list, np.ndarray))
                              else float(x[1]),
                reverse=True
            )[1:3]
          except Exception as e:
            print(f"❌ Erreur pendant le tri des similarités : {e}")
            for i, score in sim_scores:
                print(f"   ↪️ Index={i}, Type={type(score)}, Valeur={score}, Shape={getattr(score, 'shape', 'N/A')}")
            sim_scores = []


          for idx_score in sim_scores:
            name = food.iloc[idx_score[0]]['Name']
            score = idx_score[1]
            print(f"🔍 {name} | Similarité: {score:.3f}")

          content_recs.extend([food.iloc[i[0]]['Name'] for i in sim_scores])
      else:
           print(f"❌ Plat NON trouvé : '{plat_clean}'")

    # --- Collaborative-Based
    collab_recs = []
    try:
        user_id = int(user_id)
    except ValueError:
        print(f"❌ ID utilisateur invalide : {user_id}")
        user_id = None

    if user_id is not None and user_id in rating_matrix.index:
        print(f"✅ [Collaboratif] User {user_id} trouvé")
        user_ratings = ratings[ratings['User_ID'] == user_id]

        if len(user_ratings) >= 3:
            user_idx = np.where(rating_matrix.index == user_id)[0][0]
            distances, indices_knn = knn.kneighbors(
                csr_matrix(rating_matrix.iloc[user_idx].values).reshape(1, -1),
                n_neighbors=5
            )

            similar_users = rating_matrix.iloc[indices_knn[0][1:]]
            user_rated = set(user_ratings['Food_ID'])

            for neighbor_id in similar_users.index:
                neighbor_ratings = ratings[ratings['User_ID'] == neighbor_id]
                high_rated = neighbor_ratings[neighbor_ratings['Rating'] >= 7]
                unseen = high_rated[~high_rated['Food_ID'].isin(user_rated)]

                if len(unseen) >= 2:
                    top = unseen.sort_values(by="Rating", ascending=False).head(3)
                    for _, row in top.iterrows():
                        food_name = food[food['Food_ID'] == row['Food_ID']]['Name'].values
                        if len(food_name) > 0:
                            collab_recs.append(food_name[0])
                    break
        else:
            print(f"⚠️ Pas assez de notes pour l'utilisateur {user_id}")
    else:
        print(f"⚠️ Utilisateur {user_id} absent de la matrice ou ID non valide.")
    
    # --- Final merge
    hybrid = []
    seen = set()
    for rec in content_recs + collab_recs:
        if rec not in seen:
            seen.add(rec)
            hybrid.append(rec)

    return jsonify({
        "content_based": content_recs,
        "collaborative": collab_recs,
        "hybrid_recommendations": hybrid[:10]
    })



@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "working"})


