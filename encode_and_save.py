import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Charger tes plats depuis PostgreSQL ou un CSV
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT'),
    sslmode='require'
)

food = pd.read_sql(
    'SELECT id_plat AS "Food_ID", nom_plat AS "Name", categorie_plat AS "C_Type", "Description_plat" AS "Ingredient" FROM connected_restaurant."Plat"', conn
)

# Nettoyage
def text_cleaning(text):
    import string
    return "".join([char for char in text if char not in string.punctuation])

food['Ingredient'] = food['Ingredient'].apply(text_cleaning)
food['features'] = food['C_Type'].fillna('') + ' ' + food['Ingredient'].fillna('')
food['features'] = food['features'].str.strip().str.lower()

# Encoder avec BERT
print("🚀 Encodage des features avec BERT")
model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = model.encode(food['features'].tolist(), convert_to_tensor=True)

# Sauvegarde
with open("bert_embeddings.pkl", "wb") as f:
    pickle.dump(bert_embeddings, f)

food.to_pickle("food_dataframe.pkl")
print("✅ Sauvegarde terminée : bert_embeddings.pkl + food_dataframe.pkl")
