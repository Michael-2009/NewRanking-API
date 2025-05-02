from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List
import uvicorn

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class RankRequest(BaseModel):
    user_list: List[str]
    candidate_lists: List[List[str]]

@app.post("/rank_with_ids")
async def rank_lists_with_ids(data: RankRequest):
    user_name = data.user_list[0]  # optional, could be logged or returned
    user_content = data.user_list[1:]
    user_embedding = model.encode(" ".join(user_content), convert_to_tensor=True)

    ranked = []
    for candidate in data.candidate_lists:
        if not candidate or len(candidate) < 2:
            continue  # skip invalid lists
        name = candidate[0]
        content = " ".join(candidate[1:])
        cand_embedding = model.encode(content, convert_to_tensor=True)
        score = util.cos_sim(user_embedding, cand_embedding).item()
        ranked.append((name, score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
