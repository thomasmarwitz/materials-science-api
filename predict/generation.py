import pandas as pd
import re
import openai


class OpenAi:
    TITLE_PATTERN = r"\[T\]\s*(.*?)\s*\[/T\]"
    ABSTRACT_PATTERN = r"\[A\]\s*(.*?)\s*\[/A\]"

    def __init__(self, logger, api_key, organization):
        self.logger = logger
        openai.api_key = api_key
        openai.organization = organization

    def generate_abstracts(self, prompt):
        response = self._fetch_abstracts(prompt)
        self.logger.debug(f"Response: {response}")
        parsed = self._parse_response(response)
        self.logger.debug(f"Parsed: {parsed}")
        return parsed

    def _fetch_abstracts(self, prompt):
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a friendly AI agent that is an expert in the materials science domain.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=1024,
        )

    def _parse_response(self, response):
        # extract data from openai response
        data = response["choices"][0]["message"]["content"]

        titles = re.findall(self.TITLE_PATTERN, data)
        abstracts = re.findall(self.ABSTRACT_PATTERN, data)

        return [
            dict(title=title, abstract=abstract)
            for title, abstract in zip(titles, abstracts)
        ]


class Generator:
    def __init__(self, logger, graph, since, lookup_file, prompt_file, api):
        self.logger = logger
        self.logger.info("Retrieving nx graph")
        self.G = graph.get_nx_graph(since)

        self.lookup = pd.read_csv(lookup_file)
        self.prompt_template = open(prompt_file).read()
        self.api = api

    def generate_abstracts(self, conceptX, conceptY, k, min_words, max_words):
        neighborsX = self._strongest_neighbors(conceptX)
        neighborsY = self._strongest_neighbors(conceptY)

        prompt = self._format_prompt(
            topicX=conceptX,
            topicY=conceptY,
            neighborsX=neighborsX,
            neighborsY=neighborsY,
            k=k,
            min_words=min_words,
            max_words=max_words,
        )
        return dict(
            abstracts=self.api.generate_abstracts(prompt),
            neighbor_concepts=neighborsX + neighborsY,
            prompt=prompt,
        )

    def _convert(self, concept):
        return self.lookup[self.lookup["concept"] == concept].index[0]

    def _format_prompt(self, **kwargs):
        return self.prompt_template.format(**kwargs)

    def _strongest_neighbors(self, concept, k=5):
        u = self._convert(concept)

        return [
            self._translate(item)
            for item, _ in sorted(
                self.G[u].items(), key=lambda x: x[1]["links"], reverse=True
            )[:k]
        ]

    def _translate(self, u):
        return self.lookup.iloc[u]["concept"]


# Test concepts: "thermal stratification" & "biomedical alloy"
