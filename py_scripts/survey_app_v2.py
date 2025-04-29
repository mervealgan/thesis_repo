from flask import Flask, render_template, request, redirect
import sqlite3
import datetime

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def welcome():
    if request.method == "POST":
        email = request.form.get("email")
        age = request.form.get("age")
        level = request.form.get("level")
        return redirect(f"/survey?email={email}&age={age}&level={level}")
    return render_template("welcome.html")


@app.route("/survey")
def survey():
    email = request.args.get("email")
    conn = sqlite3.connect("sentences_v2.db")
    c = conn.cursor()

    c.execute("""
        SELECT id, original, simplified
        FROM allsents_v2
        WHERE votes < 5 AND id NOT IN (
            SELECT pair_id FROM voter_log WHERE email = ?
        )
        ORDER BY id ASC
    """, (email,))
    all_pairs = c.fetchall()
    pairs = [{"id": row[0], "original": row[1], "simplified": row[2]} for row in all_pairs]

    likert = [
        ("Beaucoup plus difficile à lire", -3),
        ("Plus difficile à lire", -2),
        ("Un peu plus difficile à lire", -1),
        ("Pareil", 0),
        ("Un peu plus facile à lire", 1),
        ("Plus facile à lire", 2),
        ("Beaucoup plus facile à lire", 3),
    ]


    conn.close()
    return render_template("survey_v2.html", pairs=pairs, likert=likert, email=email, age=request.args.get("age"), level=request.args.get("level"))


@app.route("/submit", methods=["POST"])
def submit():
    email = request.form.get("email")
    age = request.form.get("age")
    level = request.form.get("level")
    timestamp = datetime.datetime.now().isoformat()

    conn = sqlite3.connect("sentences_v2.db")
    c = conn.cursor()

    # For each form field
    for key in request.form:
        if key.startswith("rating_"):
            pair_id = int(key.split("_")[1])
            rating_value = int(request.form[key])

            label = next(label for label, value in [
                ("Beaucoup plus difficile", -3),
                ("Plus difficile", -2),
                ("Un peu plus difficile", -1),
                ("Pareil", 0),
                ("Un peu plus facile", 1),
                ("Plus facile", 2),
                ("Beaucoup plus facile", 3),
            ] if value == rating_value)


            # Saving to ratings_v2
            c.execute("""
                INSERT INTO ratings_v2 (pair_id, rating_label, rating_value, email, age, level, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pair_id, label, rating_value, email, age, level, timestamp))


            # Incrementing votes
            c.execute("UPDATE allsents_v2 SET votes = votes + 1 WHERE id = ?", (pair_id,))

            # Also saving to voter_log to track that this user has rated this pair
            c.execute("""
                INSERT INTO voter_log (email, pair_id)
                VALUES (?, ?)
            """, (email, pair_id))

    conn.commit()
    conn.close()


    return """
    <div style='text-align: center; font-family: Arial; margin-top: 4em;'>
        <h2>Merci ! Vos réponses ont été enregistrées.</h2>
        <a href='/' style='display: inline-block; margin-top: 2em; padding: 0.7em 1.4em; background-color: #3c7edb; color: white; text-decoration: none; border-radius: 6px;'>Je souhaite évaluer plus de phrases</a>
    </div>
    """

