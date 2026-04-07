# ============================================================
# app.py — Flask frontend with heatmap visualization
# ============================================================

import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from test.heatmap_normal import run_single_wsi_test

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Only tif and tiff allowed
ALLOWED_EXTENSIONS = {"tif", "tiff"}

MIN_PATCHES = 500

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    filename = None
    heatmap_path = None
    prob_cancer = None
    prob_non_cancer = None
    num_patches = None
    error = None

    if request.method == "POST":

        file = request.files.get("file")

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(file_path)

            result = run_single_wsi_test(
                file_path,
                output_dir=app.config["OUTPUT_FOLDER"]
            )

            num_patches = result.get("num_patches", 0)

            # Patch count validation
            if num_patches < MIN_PATCHES:

                error = (
                    f"Only {num_patches} tissue patches were detected in this slide. "
                    f"A minimum of {MIN_PATCHES} patches is required for reliable classification."
                )

            else:

                prediction = result["prediction"]
                confidence = round(result["confidence"] * 100, 2)

                prob_cancer = round(result["prob_cancer"] * 100, 2)
                prob_non_cancer = round(result["prob_non_cancer"] * 100, 2)

                if result.get("heatmap_path") and os.path.exists(result["heatmap_path"]):

                    heatmap_path = os.path.basename(result["heatmap_path"])

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        heatmap_path=heatmap_path,
        prob_cancer=prob_cancer,
        prob_non_cancer=prob_non_cancer,
        num_patches=num_patches,
        error=error,
        min_patches=MIN_PATCHES
    )


@app.route("/outputs/<path:filename>")
def serve_output(filename):

    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)