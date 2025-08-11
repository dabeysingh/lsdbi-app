import os
from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from prestodb.dbapi import connect as presto_connect
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, Model

app = Flask(__name__)

# Milvus (eu-de)
MILVUS_HOST=os.getenv("MILVUS_HOST"); MILVUS_PORT=os.getenv("MILVUS_PORT","32532")
MILVUS_USER=os.getenv("MILVUS_USER_EU"); MILVUS_PASS=os.getenv("MILVUS_PASS_EU"); MILVUS_COLLECTION=os.getenv("MILVUS_COLLECTION","evidence")

# Presto (eu-de)
PRESTO_HOST=os.getenv("PRESTO_HOST"); PRESTO_PORT=int(os.getenv("PRESTO_PORT","32700"))
PRESTO_CATALOG=os.getenv("PRESTO_CATALOG","lsdbi"); PRESTO_SCHEMA=os.getenv("PRESTO_SCHEMA","lsdbi-v2")
PRESTO_USER=os.getenv("PRESTO_USER","code-engine"); PRESTO_TLS=os.getenv("PRESTO_TLS","true").lower()=="true"

# watsonx.ai (us-south)
WX_APIKEY=os.getenv("WX_APIKEY"); WX_URL=os.getenv("WX_URL","https://us-south.ml.cloud.ibm.com")
WX_PROJECT_ID=os.getenv("WX_PROJECT_ID")
EMB_MODEL_ID=os.getenv("EMB_MODEL_ID","ibm/slate-embedding-uncased")
GEN_MODEL_ID=os.getenv("GEN_MODEL_ID","granite-20b-instruct")

def presto_conn():
    return presto_connect(host=PRESTO_HOST, port=PRESTO_PORT, user=PRESTO_USER,
                          catalog=PRESTO_CATALOG, schema=PRESTO_SCHEMA,
                          http_scheme="https" if PRESTO_TLS else "http")

def milvus_col():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT, secure=True, token=f"{MILVUS_USER}:{MILVUS_PASS}")
    return Collection(MILVUS_COLLECTION)

creds = Credentials(api_key=WX_APIKEY, url=WX_URL)
embedder = Embeddings(model_id=EMB_MODEL_ID, credentials=creds, project_id=WX_PROJECT_ID)
generator = Model(model_id=GEN_MODEL_ID, credentials=creds, project_id=WX_PROJECT_ID,
                  params={"decoding_method":"greedy","max_new_tokens":600,"temperature":0})

def embed(text):
    return embedder.embed([text])["results"][0]["embedding"]

def asof_to_days(s):
    import datetime
    if not s: return None
    y,m,d = map(int, s.split("-"))
    return datetime.date(y,m,d).toordinal()

@app.route("/search", methods=["POST"])
def search():
    p = request.get_json(force=True)
    q = p.get("query",""); division=p.get("division"); sport=p.get("sport"); as_of=p.get("as_of"); k=int(p.get("k",8))
    vec = [embed(q)]
    exprs=[]
    if division: exprs.append(f'division == "{division}"')
    if as_of:
        days = asof_to_days(as_of)
        exprs.append(f'effective_start_days <= {days} && (effective_end_days == 0 || {days} < effective_end_days)')
    expr = " && ".join(exprs) if exprs else ""
    col = milvus_col()
    out = col.search(data=vec, anns_field="embedding",
                     param={"metric_type":"COSINE","params":{"ef":64}},
                     limit=k, expr=expr,
                     output_fields=["evidence_id","division","bylaw_display","bylaw_number","part_code",
                                    "effective_start_days","effective_end_days","span_text"])
    hits=[]
    if out:
        for h in out[0]:
            r=h.entity
            hits.append({
              "evidence_id": r.get("evidence_id"), "division": r.get("division"),
              "bylaw_display": r.get("bylaw_display"), "bylaw_number": r.get("bylaw_number"),
              "part_code": r.get("part_code"),
              "effective_start": r.get("effective_start_days"),
              "effective_end": r.get("effective_end_days"),
              "span_text": r.get("span_text"),
              "score": float(h.distance)
            })
    return jsonify({"hits": hits})

@app.route("/sql", methods=["POST"])
def nl2sql():
    p=request.get_json(force=True); question=p.get("question",""); as_of=p.get("as_of")
    schema_prompt = """
You generate SQL for Presto (watsonx.data) against ONLY these views:
- v_bylaw_current(...)
- v_bylaw_part_current(...)
- v_interp_current(...)
- v_bylaw_all_with_lineage(...)
Rules:
- If AS_OF is present, filter rows where effective_date <= AS_OF < COALESCE(discontinue_date, DATE '9999-12-31').
- Prefer v_bylaw_part_current for citeable text; use v_bylaw_all_with_lineage to map old->current numbers.
Return SQL only.
""".strip()
    prompt = f"{schema_prompt}\nQuestion: {question}\nAS_OF: {as_of or 'TODAY'}\nSQL:"
    sql_text = generator.generate_text(prompt=prompt)["results"][0]["generated_text"].strip()
    con = presto_conn(); cur=con.cursor()
    try:
        cur.execute(sql_text); rows=cur.fetchall(); cols=[d[0] for d in cur.description]
        return jsonify({"rows":[dict(zip(cols,r)) for r in rows], "sql":sql_text, "rowcount":len(rows)})
    except Exception as e:
        return jsonify({"error":str(e), "sql":sql_text}), 400

@app.route("/format", methods=["POST"])
def format_answer():
    p=request.get_json(force=True); question=p.get("question",""); ev=p.get("evidence",[])
    sys=("Answer using ONLY evidence. Cite like [Bylaw <display>(<part>), <division>, eff. <YYYY-MM-DD>] or "
         "[Interp <id>, <division>, eff. <YYYY-MM-DD>]. If divisions differ, split bullets.")
    evtxt="\n".join(f"- {e.get('span_text','')}\n  [src: {e.get('bylaw_display') or ''} {e.get('part_code') or ''}, {e.get('division') or ''}]"
                    for e in ev[:10])
    out = generator.generate_text(prompt=f"{sys}\n\nQuestion: {question}\n\nEvidence:\n{evtxt}\n\nAnswer:")
    return jsonify({"answer": out['results'][0]['generated_text'].strip()})

@app.route("/healthz")
def healthz(): return jsonify({"ok":True})
