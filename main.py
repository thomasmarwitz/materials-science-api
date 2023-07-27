from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search.search import sem_search, plain_search

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search")
def search(query: str, semantic: bool = False, k: int = 10):
    print(f"Searching term {query}")
    if semantic:
        return sem_search.search(query, k=k)
    else:
        return plain_search.search(query, k=k)
