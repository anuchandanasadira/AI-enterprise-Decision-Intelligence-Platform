from flask import Flask, render_template, request
from rag import retrieve_documents
from agents import run_multi_agent_analysis, decision_agent

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""
    agent_results = None

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        if question:

            print("\nQuestion:", question)

            # RAG retrieval
            retrieved_documents = retrieve_documents(
                question,
                k=5
            )

            # Multi-Agent AI
            agent_results = run_multi_agent_analysis(
                question
            )

            # Decision Agent
            answer = decision_agent(
                question,
                agent_results
            )

    return render_template(
        "dashboard.html",
        question=question,
        answer=answer,
        agent_results=agent_results
    )


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False
    )