from flask import Flask, render_template, request
from phisDetectorWeb import featureExtraction, fuzzy_score, compute_percentage

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    score = None
    percentage_score = None
    url = ''
    breakdown = {}

    if request.method == 'POST':
        url = request.form['url']
        features = featureExtraction(url)

        if not features.empty:
            score, breakdown = fuzzy_score(features.iloc[0], url)
            percentage_score = compute_percentage(score)
            if score > 0.6:
                result = "URL ini terlihat mencurigakan."
            elif score > 0.3:
                result = "URL ini mungkin perlu diperiksa lebih lanjut."
            else:
                result = "URL ini tampaknya aman."

    return render_template('index.html', url=url, score=score, percentage_score=percentage_score, result=result, breakdown=breakdown)

if __name__ == '__main__':
    app.run(debug=True)
