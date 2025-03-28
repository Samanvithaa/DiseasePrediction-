async function predictDisease() {
    let symptoms = document.getElementById("symptoms").value;
    if (!symptoms) {
        alert("Please enter symptoms");
        return;
    }

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: symptoms })
    });

    let data = await response.json();
    if (data.error) {
        document.getElementById("result").innerText = "Error: " + data.error;
    } else {
        document.getElementById("result").innerText =
            `Predicted Disease: ${data.disease} \nMedical Test: ${data.test} \nCost: ₹${data.cost}`;
    }
}
