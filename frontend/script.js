const form = document.getElementById("predictionForm");

const result = document.getElementById("result");

const probabilityElement =
    document.getElementById("probability");

const predictionElement =
    document.getElementById("prediction");

const thresholdElement =
    document.getElementById("threshold");

const button =
    document.getElementById("predictButton");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    button.disabled = true;
    button.textContent = "Predicting...";


    const data = {

        Administrative:
            Number(
                document.getElementById(
                    "Administrative"
                ).value
            ),

        Administrative_Duration:
            Number(
                document.getElementById(
                    "Administrative_Duration"
                ).value
            ),

        Informational:
            Number(
                document.getElementById(
                    "Informational"
                ).value
            ),

        Informational_Duration:
            Number(
                document.getElementById(
                    "Informational_Duration"
                ).value
            ),

        ProductRelated:
            Number(
                document.getElementById(
                    "ProductRelated"
                ).value
            ),

        ProductRelated_Duration:
            Number(
                document.getElementById(
                    "ProductRelated_Duration"
                ).value
            ),

        BounceRates:
            Number(
                document.getElementById(
                    "BounceRates"
                ).value
            ),

        ExitRates:
            Number(
                document.getElementById(
                    "ExitRates"
                ).value
            ),

        PageValues:
            Number(
                document.getElementById(
                    "PageValues"
                ).value
            ),

        SpecialDay:
            Number(
                document.getElementById(
                    "SpecialDay"
                ).value
            ),

        Month:
            document.getElementById(
                "Month"
            ).value,

        OperatingSystems:
            Number(
                document.getElementById(
                    "OperatingSystems"
                ).value
            ),

        Browser:
            Number(
                document.getElementById(
                    "Browser"
                ).value
            ),

        Region:
            Number(
                document.getElementById(
                    "Region"
                ).value
            ),

        TrafficType:
            Number(
                document.getElementById(
                    "TrafficType"
                ).value
            ),

        VisitorType:
            document.getElementById(
                "VisitorType"
            ).value,

        Weekend:
            document.getElementById(
                "Weekend"
            ).checked
    };


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        if (!response.ok) {
            throw new Error(
                "Prediction request failed"
            );
        }


        const prediction =
            await response.json();


        const probability =
            prediction.purchase_probability * 100;


        probabilityElement.textContent =
            probability.toFixed(2) + "%";


        thresholdElement.textContent =
            prediction.threshold;


        predictionElement.className =
            "prediction";


        if (prediction.will_purchase) {

            predictionElement.textContent =
                "Likely to Purchase";

            predictionElement.classList.add(
                "purchase"
            );

        } else {

            predictionElement.textContent =
                "Unlikely to Purchase";

            predictionElement.classList.add(
                "no-purchase"
            );
        }


        result.classList.remove(
            "hidden"
        );


        result.scrollIntoView({
            behavior: "smooth"
        });


    } catch (error) {

        alert(
            "Unable to get prediction. " +
            "Make sure the API is running."
        );

        console.error(error);

    } finally {

        button.disabled = false;

        button.textContent =
            "Predict Purchase";
    }

});