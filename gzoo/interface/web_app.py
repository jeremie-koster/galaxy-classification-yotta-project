"""
    Create the web app to expose the model
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gzoo.app.predict import main as predict_pipeline

# Constants
INFERENCE_FOLDER = "gzoo/interface/inference_webapp/"
IMAGE_DEST_FOLDER = os.path.join(INFERENCE_FOLDER, "images_test_rev1/")
PREDICTIONS_FOLDER = os.path.join(INFERENCE_FOLDER, "predictions/")
IMAGE_EXTENSIONS = [".jpg", ".png"]
IMAGE_NEEDED_FOLDER = os.path.join("gzoo/interface/", "images_needed/")
HUBBLE = os.path.join(IMAGE_NEEDED_FOLDER, "hubble.png")


def save_image(image):
    """ Save the image provided by the user in order to run the prediction pipeline """
    image.save(os.path.join(IMAGE_DEST_FOLDER, "user_image.jpg"))


def load_image(image_file):
    """ Load the image provided by the user """
    img = Image.open(image_file)
    img.thumbnail((400, 400))
    return img


def remove_image(folder):
    """ Remove any image from the inference folder (once the prediction is done) """
    for file in os.listdir(folder):
        if file.endswith(tuple(IMAGE_EXTENSIONS)):
            os.remove(os.path.join(folder, file))


def load_prediction_results(folder):
    """ Retrieve the results of the prediction """
    results_path = os.path.join(folder, "predictions.csv")
    results = pd.read_csv(results_path)
    return results


def main():

    html_temp = """
    <div style="background:#2176ae ;padding:10px">
    <h2 style="color:white;text-align:center;"> Galaxy morphology prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("##  ")

    image_file = st.sidebar.file_uploader(
        "Upload your galaxy image", IMAGE_EXTENSIONS, key="uploader_key", help="This is helpful"
    )

    if image_file is not None:
        image = load_image(image_file)
        save_image(image)
        st.image(image, caption="Your image")

    if st.sidebar.button("Predict"):
        with st.spinner("Making the prediction..."):
            predict_pipeline(yaml_path="config/predict_webapp.yaml", run_cli=False)
        st.sidebar.success("Prediction done!")
        prediction_results = load_prediction_results(PREDICTIONS_FOLDER)
        results = prediction_results[
            [
                "completely_round_smooth",
                "in_between_smooth",
                "cigar_shaped_smooth",
                "edge_on",
                "spiral",
            ]
        ]

        pred = np.array(prediction_results)
        class_max_proba = prediction_results.columns

        max_proba = prediction_results.max(axis=1)
        max_proba = pred[pred == max_proba[0]]
        cond = np.where(pred == max_proba)[1][0]
        class_max_proba = class_max_proba[cond]

        plt.rcdefaults()
        fig, ax = plt.subplots()
        col = results.columns.to_numpy().flatten()
        proba = results.to_numpy().flatten()
        y_pos = np.arange(len(col))

        ax.barh(y_pos, proba, align="center", color="red")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(col)
        ax.invert_yaxis()
        ax.set_xlabel("Probabilities")
        ax.set_title("Classes Probabilities")

        st.pyplot(fig)

        if class_max_proba == "spiral":

            st.write(f"Your Galaxy has a high probability to be a Spiral galaxy.")
            st.write(
                f"Spiral class represents SBa, SBb, SBc, Sa, Sb, Sc classes in Hubble classification."
            )

            col1, col2 = st.beta_columns(2)

            hubble = Image.open(HUBBLE)
            col1.image(hubble, use_column_width=True)

            spirale = Image.open(os.path.join(IMAGE_NEEDED_FOLDER, "spiral.jpg"))
            col2.image(spirale, use_column_width=True)

        elif class_max_proba == "completely_round_smooth":

            st.write(f"Your Galaxy has a high probability to be a Completely Round Smooth galaxy.")
            st.write(
                f"Completely round smooth class represents E0, E1 classes in Hubble classification."
            )

            col1, col2 = st.beta_columns(2)

            hubble = Image.open(HUBBLE)
            col1.image(hubble, use_column_width=True)

            round_smooth = Image.open(os.path.join(IMAGE_NEEDED_FOLDER, "elliptical.jpg"))
            col2.image(round_smooth, use_column_width=True)

        elif class_max_proba == "in_between_smooth":

            st.write(f"Your Galaxy has a high probability to be an In Between Smooth galaxy.")
            st.write(
                f"In between smooth class represents E2, E3, E4, E5, E6 classes in Hubble classification."
            )

            col1, col2 = st.beta_columns(2)

            hubble = Image.open(HUBBLE)
            col1.image(hubble, use_column_width=True)

            in_between_smooth = Image.open(
                os.path.join(IMAGE_NEEDED_FOLDER, "in_between_smooth.jpg")
            )
            col2.image(in_between_smooth, use_column_width=True)

        elif class_max_proba == "cigar_smooth":

            st.write(f"Your Galaxy has a high probability to be a Cigar Smooth galaxy.")
            st.write(
                f"Completely round smooth class represents E7 classes in Hubble classification. E7 is the most elongated galaxy of the Elliptical galaxies"
            )

            col1, col2 = st.beta_columns(2)

            hubble = Image.open(HUBBLE)
            col1.image(hubble, use_column_width=True)

            cigar_smooth = Image.open(os.path.join(IMAGE_NEEDED_FOLDER, "cigar.jpeg"))
            col2.image(cigar_smooth, use_column_width=True)

        else:
            st.write(f"Your Galaxy has a high probability to be an Edge On galaxy.")
            st.write(f"Edge On class means all classes in Hubble classification.")

            st.image(HUBBLE)

        remove_image(IMAGE_DEST_FOLDER)


if __name__ == "__main__":
    main()
