from .graph import Graph
import pandas as pd
from ast import literal_eval
import re
import os
from dotenv import load_dotenv

load_dotenv()

print("Loading graph")
G = Graph.from_path("data/graph/edges.M.pkl").get_nx_graph(2019)
lookup = pd.read_csv(
    "data/lookup/lookup.M.new.csv",
)
PROMPT_TEMPLATE = open("data/prompt.txt").read()


def connecting_works(u, v, k=5):
    u_works = literal_eval(lookup.iloc[u]["works"])
    v_works = literal_eval(lookup.iloc[v]["works"])
    return [*(set(u_works) & set(v_works))][:k]


def convert(concept):
    return lookup[lookup["concept"] == concept].index[0]


def translate(u):
    return lookup.iloc[u]["concept"]


def common_neighbors(u, v):
    return [translate(item) for item in set(G.neighbors(u)) & set(G.neighbors(v))]


def strongest_neighbors(u, k=5):
    return [
        translate(item)
        for item, _ in sorted(G[u].items(), key=lambda x: x[1]["links"], reverse=True)[
            :k
        ]
    ]


def format_prompt(**kwargs):
    return PROMPT_TEMPLATE.format(**kwargs)


def gen_prompt(conceptX, conceptY, k=5):
    u = convert(conceptX)
    v = convert(conceptY)

    return format_prompt(
        topicX=conceptX,
        topicY=conceptY,
        neighborsX=strongest_neighbors(u),
        neighborsY=strongest_neighbors(v),
        k=k,
    )


import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


def fetch_abstracts(conceptX, conceptY, k=5):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a friendly AI agent that is an expert in the materials science domain.",
            },
            {
                "role": "user",
                "content": gen_prompt(conceptX, conceptY, k=k),
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )


title_pattern = r"\[T\]\s*(.*?)\s*\[/T\]"
abstract_pattern = r"\[A\]\s*(.*?)\s*\[/A\]"


def parse_response(response):
    # extract data from openai response
    data = response["choices"][0]["message"]["content"]

    titles = re.findall(title_pattern, data)
    abstracts = re.findall(abstract_pattern, data)

    return [
        dict(title=title, abstract=abstract)
        for title, abstract in zip(titles, abstracts)
    ]


def generate_abstracts(conceptX, conceptY, k=5):
    response = fetch_abstracts(conceptX, conceptY, k=k)
    return parse_response(response)


# Test concepts: "thermal stratification" & "biomedical alloy"
