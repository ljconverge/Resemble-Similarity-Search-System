from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("documents")
        threshold = request.form.get("threshold", "80")
        uploaded_names = [f.filename for f in files if f.filename]
        return redirect(url_for("results", threshold=threshold, count=len(uploaded_names)))
    return render_template("index.html")

@app.route("/results")
def results():
    threshold = request.args.get("threshold", "80")
    count = request.args.get("count", "4")
    return render_template("results.html", threshold=threshold, count=count)

@app.route("/performance")
def performance():
    return render_template("performance.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(debug=True)
