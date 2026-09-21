document.getElementById("wineForm").addEventListener("submit", async function(event) {

    event.preventDefault();

    const data = {
        fixed_acidity: parseFloat(document.getElementById("fixed_acidity").value),
        volatile_acidity: parseFloat(document.getElementById("volatile_acidity").value),
        citric_acid: parseFloat(document.getElementById("citric_acid").value),
        residual_sugar: parseFloat(document.getElementById("residual_sugar").value),
        chlorides: parseFloat(document.getElementById("chlorides").value),
        free_sulfur_dioxide: parseFloat(document.getElementById("free_sulfur_dioxide").value),
        total_sulfur_dioxide: parseFloat(document.getElementById("total_sulfur_dioxide").value),
        density: parseFloat(document.getElementById("density").value),
        pH: parseFloat(document.getElementById("pH").value),
        sulphates: parseFloat(document.getElementById("sulphates").value),
        alcohol: parseFloat(document.getElementById("alcohol").value)
    };

    document.getElementById("result").textContent = "Predicting...";

    try {

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("API request failed");
        }

        const result = await response.json();

        document.getElementById("result").textContent = result.result;

    } catch (error) {

        console.error(error);

        document.getElementById("result").textContent =
            "Backend connection error!";
    }

});