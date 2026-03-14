from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")
pairs = [["What was the revenue in Q4?", "Q4 revenue was 15."]]
scores = reranker.predict(pairs)
print("Scores:", scores)
print("Type:", type(scores))

try:
    for r, score in zip([{"content": "a"}], scores):
        print("Zip worked:", score)
except Exception as e:
    print("Zip failed!", e)
