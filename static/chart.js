// Inventory Risk Pie Chart
const inventoryData = window.inventoryData || [];

let highRisk = 0;
let mediumRisk = 0;
let lowRisk = 0;

inventoryData.forEach(product => {
    if (product.risk === "HIGH") {
        highRisk++;
    } else if (product.risk === "MEDIUM") {
        mediumRisk++;
    } else {
        lowRisk++;
    }
});

const inventoryCanvas = document.getElementById("inventoryChart");

if (inventoryCanvas) {
    new Chart(inventoryCanvas, {
        type: "pie",
        data: {
            labels: ["High Risk", "Medium Risk", "Low Risk"],
            datasets: [{
                data: [highRisk, mediumRisk, lowRisk]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    });
}


// Customer Sentiment Pie Chart
const reviewData = window.reviewData || {};

const negativeReviews = reviewData.negative_reviews || 0;
const otherReviews = reviewData.other_reviews || 0;

const reviewCanvas = document.getElementById("reviewChart");

if (reviewCanvas) {
    new Chart(reviewCanvas, {
        type: "pie",
        data: {
            labels: ["Negative Reviews", "Other Reviews"],
            datasets: [{
                data: [negativeReviews, otherReviews]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    });
}