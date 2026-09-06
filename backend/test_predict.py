from PIL import Image
from predict_v3 import predict_image


test_images = [
    "../test_images/car.jpg",
    "../test_images/dog.jpg",
    "../test_images/industry_ro.png",
    "../test_images/Forest_537.jpg",
    "../test_images/Highway_1325.jpg"
]


for image_path in test_images:

    print("\n" + "=" * 55)
    print("IMAGE:", image_path)
    print("=" * 55)

    image = Image.open(image_path)

    result = predict_image(image)

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")

    print("\nTop 3:")

    for item in result["top3"]:
        print(
            f"  {item['class']}: "
            f"{item['probability']}%"
        )

    print("\n--- Uncertainty ---")

    print( "Entropy:",result["entropy"] )

    print("Margin:",result["margin"],"%")

    print("\n--- Domain Analysis ---")

    print( "Feature similarity:",result["feature_similarity"])

    print( "Nearest EuroSAT class:",result["nearest_eurosat_class"] )

    print("OOD threshold:",result["ood_threshold"])
    print("Possible OOD:",result["ood"])

    print("\n--- Reliability ---")

    print("Reliability score:",result["reliability_score"],"/ 100")

    print("Reliability status:",result["reliability_status"])