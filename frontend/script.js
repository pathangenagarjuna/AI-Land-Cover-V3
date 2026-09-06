const API_URL = "https://ai-land-cover-v3.vercel.app/api";


const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const chooseButton = document.getElementById("chooseButton");
const analyzeButton = document.getElementById("analyzeButton");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const removeButton = document.getElementById("removeButton");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const results = document.getElementById("results");

let selectedFile = null;


/* =========================================================
   Choose file
========================================================= */

chooseButton.addEventListener(
    "click",
    () => {
        fileInput.click();
    }
);


fileInput.addEventListener(
    "change",
    () => {

        if (fileInput.files.length > 0) {
            setFile(fileInput.files[0]);
        }

    }
);


/* =========================================================
   Drag & Drop
========================================================= */

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {
            setFile(files[0]);
        }

    }
);


/* =========================================================
   Set selected file
========================================================= */

function setFile(file) {

    if (!file.type.startsWith("image/")) {

        showError(
            "Please select a JPG, JPEG or PNG image."
        );

        return;
    }


    selectedFile = file;


    previewImage.src =
        URL.createObjectURL(file);

    fileName.textContent =
        file.name;


    previewContainer.classList.remove(
        "hidden"
    );


    analyzeButton.disabled = false;


    results.classList.add(
        "hidden"
    );


    hideError();
}


/* =========================================================
   Remove image
========================================================= */

removeButton.addEventListener(
    "click",
    () => {

        selectedFile = null;

        fileInput.value = "";

        previewContainer.classList.add(
            "hidden"
        );

        analyzeButton.disabled = true;

        results.classList.add(
            "hidden"
        );

        hideError();

    }
);


/* =========================================================
   Analyze
========================================================= */

analyzeButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {
            return;
        }


        const formData =
            new FormData();

        formData.append(
            "file",
            selectedFile
        );


        setLoading(true);


        try {

            const response =
                await fetch(
                    `${API_URL}/predict`,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Prediction request failed."
                );

            }


            const data =
                await response.json();


            displayResults(data);


        } catch (error) {

            console.error(error);

            showError(
                "Could not connect to the AI backend. Make sure FastAPI is running."
            );

        } finally {

            setLoading(false);

        }

    }
);


/* =========================================================
   Display results
========================================================= */

function displayResults(data) {

    results.classList.remove(
        "hidden"
    );


    /* =====================================================
       Main prediction
    ===================================================== */

    document.getElementById(
        "prediction"
    ).textContent =
        data.prediction;


    document.getElementById(
        "confidence"
    ).textContent =
        data.confidence;


    /* =====================================================
       Uncertainty
    ===================================================== */

    document.getElementById(
        "entropy"
    ).textContent =
        data.entropy;


    document.getElementById(
        "margin"
    ).textContent =
        `${data.margin}%`;


    /* =====================================================
       Domain analysis
    ===================================================== */

    document.getElementById(
        "featureSimilarity"
    ).textContent =
        data.feature_similarity;


    document.getElementById(
        "oodThreshold"
    ).textContent =
        data.ood_threshold;


    document.getElementById(
        "nearestClass"
    ).textContent =
        data.nearest_eurosat_class;


    /* =====================================================
       Domain fit
    ===================================================== */

    document.getElementById(
        "reliabilityScore"
    ).textContent =
        `${data.reliability_score} / 100`;


    document.getElementById(
        "domainFit"
    ).textContent =
        `${data.reliability_score} / 100`;


    document.getElementById(
        "reliabilityStatus"
    ).textContent =
        data.reliability_status;


    /* =====================================================
       OOD status
    ===================================================== */

    const oodCard =
        document.getElementById(
            "oodCard"
        );

    const oodStatus =
        document.getElementById(
            "oodStatus"
        );


    if (data.ood) {

        oodCard.classList.add(
            "warning"
        );

        oodStatus.textContent =
            "⚠ Possible OOD — prediction may be unreliable";

    } else {

        oodCard.classList.remove(
            "warning"
        );

        oodStatus.textContent =
            "✓ Looks like in-domain imagery";

    }


    /* =====================================================
       Top 3
    ===================================================== */

    const top3Container =
        document.getElementById(
            "top3"
        );


    top3Container.innerHTML = "";


    data.top3.forEach(
        (item) => {

            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "top3-row";


            row.innerHTML = `

                <div class="top3-header">

                    <span>
                        ${item.class}
                    </span>

                    <strong>
                        ${item.probability}%
                    </strong>

                </div>

                <div class="progress">

                    <div
                        class="progress-bar"
                        style="width: ${item.probability}%"
                    ></div>

                </div>

            `;


            top3Container.appendChild(
                row
            );

        }
    );

}


/* =========================================================
   Loading
========================================================= */

function setLoading(isLoading) {

    if (isLoading) {

        loading.classList.remove(
            "hidden"
        );

        analyzeButton.disabled = true;

    } else {

        loading.classList.add(
            "hidden"
        );

        analyzeButton.disabled =
            !selectedFile;

    }

}


/* =========================================================
   Errors
========================================================= */

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

}


function hideError() {

    errorBox.classList.add(
        "hidden"
    );

}