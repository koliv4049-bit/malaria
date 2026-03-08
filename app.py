from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("malaria_model.h5")

@app.route("/", methods=["GET","POST"])
def index():

    result=""

    if request.method=="POST":

        file = request.files["file"]
        file.save("test.png")

        img = image.load_img("test.png", target_size=(64,64))
        img = image.img_to_array(img)/255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)

        if pred[0][0] > 0.5:
            result="Malaria Detected"
        else:
            result="Healthy Cell"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)