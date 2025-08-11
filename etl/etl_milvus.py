import os, pandas as pd
from tqdm import tqdm
from prestodb.dbapi import connect as presto_connect
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# --- Env (eu-de Presto & Milvus; us-south watsonx.ai) ---
PRESTO_HOST   = os.getenv("PRESTO_HOST")
PRESTO_PORT   = int(os.getenv("PRESTO_PORT", "32700"))
PRESTO_CATALOG= os.getenv("PRESTO_CATALOG", "lsdbi")
PRESTO_SCHEMA = os.getenv("PRESTO_SCHEMA", "lsdbi-v2")
PRESTO_USER   = os.getenv("PRESTO_USER", "code-engine")
PRESTO_TLS    = os.getenv("PRESTO_TLS", "true").lower() == "true"

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT", "32532")
MILVUS_USER = os.getenv("MILVUS_USER_EU")
MILVUS_PASS = os.getenv("MILVUS_PASS_EU")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "evidence")
PARTITION_KEY_FIELD = "sport_pk"

WX_APIKEY     = os.getenv("WX_APIKEY")
WX_URL        = os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
EMB_MODEL_ID  = os.getenv("EMB_MODEL_ID", "ibm/slate-embedding-uncased")
EMB_DIM       = int(os.getenv("EMB_DIM", "1024"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "128"))
LIMIT_ROWS    = os.getenv("LIMIT_ROWS")

def presto_conn():
    return presto_connect(
        host=PRESTO_HOST, port=PRESTO_PORT, user=PRESTO_USER,
        catalog=PRESTO_CATALOG, schema=PRESTO_SCHEMA,
        http_scheme="https" if PRESTO_TLS else "http"
    )

def fetch_evidence_df():
    sql = f"""
    SELECT
      evidence_id, source_type, source_id, division, sport_codes,
      bylaw_id, bylaw_display, bylaw_number, part_code, title,
      effective_start, effective_end,
      ARRAY_REMOVE(ARRAY[level1,level2,level3,level4,level5,level6,level7,level8], NULL) AS topic_path,
      span_text
    FROM "{PRESTO_CATALOG}"."{PRESTO_SCHEMA}".v_evidence_unit
    """
    if LIMIT_ROWS: sql += f" LIMIT {int(LIMIT_ROWS)}"
    con = presto_conn()
    return pd.read_sql(sql, con)

def connect_milvus():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, secure=True, token=f"{MILVUS_USER}:{MILVUS_PASS}")

def ensure_collection():
    if utility.has_collection(MILVUS_COLLECTION):
        col = Collection(MILVUS_COLLECTION); col.load(); return col
    fields = [
        FieldSchema(name="evidence_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="division", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name=PARTITION_KEY_FIELD, dtype=DataType.VARCHAR, max_length=32, is_partition_key=True),
        FieldSchema(name="bylaw_id", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="bylaw_display", dtype=DataType.VARCHAR, max_length=64, is_nullable=True),
        FieldSchema(name="bylaw_number", dtype=DataType.VARCHAR, max_length=64, is_nullable=True),
        FieldSchema(name="part_code", dtype=DataType.VARCHAR, max_length=64, is_nullable=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512, is_nullable=True),
        FieldSchema(name="effective_start_days", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="effective_end_days", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="topic_l1", dtype=DataType.VARCHAR, max_length=128, is_nullable=True),
        FieldSchema(name="topic_l2", dtype=DataType.VARCHAR, max_length=128, is_nullable=True),
        FieldSchema(name="topic_l3", dtype=DataType.VARCHAR, max_length=128, is_nullable=True),
        FieldSchema(name="span_text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM),
    ]
    schema = CollectionSchema(fields, description="LSDBI evidence spans")
    col = Collection(name=MILVUS_COLLECTION, schema=schema)
    col.create_index(field_name="embedding", index_params={"index_type":"HNSW","metric_type":"COSINE","params":{"M":16,"efConstruction":200}})
    col.load()
    return col

def epoch_days(d):
    if d is None or pd.isna(d): return 0
    if isinstance(d, str): d = pd.to_datetime(d)
    return int(d.to_pydatetime().date().toordinal())

def first_sport(sport_codes):
    return (str(sport_codes[0])[:32] if isinstance(sport_codes, list) and len(sport_codes) else "NONE")

def top_topics(topic_path):
    if not isinstance(topic_path, list): return (None, None, None)
    vals = [t for t in topic_path if t]
    return (vals[0] if len(vals)>0 else None, vals[1] if len(vals)>1 else None, vals[2] if len(vals)>2 else None)

def embedder():
    return Embeddings(model_id=EMB_MODEL_ID, credentials=Credentials(api_key=WX_APIKEY, url=WX_URL), project_id=WX_PROJECT_ID)

def main():
    print("Fetching evidence …")
    df = fetch_evidence_df()
    if df.empty:
        print("No evidence rows."); return

    print("Connecting Milvus …")
    connect_milvus(); col = ensure_collection()

    emb = embedder()
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        sl = df.iloc[i:i+BATCH_SIZE].copy()
        vectors = [r["embedding"] for r in emb.embed(sl["span_text"].astype(str).tolist())["results"]]
        ents = {
            "evidence_id": sl["evidence_id"].astype(str).tolist(),
            "division": sl["division"].astype(str).tolist(),
            "sport_pk": [first_sport(x) for x in sl["sport_codes"].tolist()],
            "bylaw_id": [int(x) if pd.notna(x) else None for x in sl["bylaw_id"].tolist()],
            "bylaw_display": sl["bylaw_display"].astype(str).where(sl["bylaw_display"].notna(), None).tolist(),
            "bylaw_number": sl["bylaw_number"].astype(str).where(sl["bylaw_number"].notna(), None).tolist(),
            "part_code": sl["part_code"].astype(str).where(sl["part_code"].notna(), None).tolist(),
            "title": sl["title"].astype(str).where(sl["title"].notna(), None).tolist(),
            "effective_start_days": [epoch_days(x) for x in sl["effective_start"].tolist()],
            "effective_end_days": [epoch_days(x) for x in sl["effective_end"].tolist()],
            "topic_l1": [], "topic_l2": [], "topic_l3": [],
            "span_text": sl["span_text"].astype(str).tolist(),
            "embedding": vectors,
        }
        for tp in sl["topic_path"].tolist():
            l1,l2,l3 = top_topics(tp); ents["topic_l1"].append(l1); ents["topic_l2"].append(l2); ents["topic_l3"].append(l3)
        data = [ents[k] for k in ["evidence_id","division","sport_pk","bylaw_id","bylaw_display","bylaw_number",
                                  "part_code","title","effective_start_days","effective_end_days",
                                  "topic_l1","topic_l2","topic_l3","span_text","embedding"]]
        col.upsert(data)
    col.flush()
    print("Done.")
    connections.disconnect("default")

if __name__ == "__main__":
    main()
