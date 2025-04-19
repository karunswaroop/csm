import lmstudio as lms

with lms.Client() as client:
    model = client.llm.model("gemma-3-4b-it-qat")
    result = model.respond("What is the meaning of life?")

    print(result)

