let priceChart = null;

async function calculatePrice() {
    // Show loading indicator
    document.getElementById('resultsContainer').classList.add('hidden');
    document.getElementById('errorContainer').classList.add('hidden');
    document.getElementById('loadingIndicator').classList.remove('hidden');

    try {
        // Get form data
        const customerData = {
            Recency: parseFloat(document.getElementById('Recency').value),
            Frequency: parseFloat(document.getElementById('Frequency').value),
            MonetaryValue: parseFloat(document.getElementById('MonetaryValue').value),
            Tenure: parseFloat(document.getElementById('Tenure').value),
            AvgDaysBetweenPurchases: parseFloat(document.getElementById('AvgDaysBetweenPurchases').value),
            Age: parseFloat(document.getElementById('Age').value),
            UniqueProductsCount: parseFloat(document.getElementById('UniqueProductsCount').value),
            product_cost: parseFloat(document.getElementById('product_cost').value)
        };

        // Call API
        const response = await fetch('/api/calculate_price/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData)
        });

        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        // Display results
        displayResults(data.data);
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

async function testModel() {
    try {
        document.getElementById('loadingIndicator').classList.remove('hidden');
        
        const response = await fetch('/api/test_model/');
        const data = await response.json();

        if (data.status === 'error') {
            throw new Error(data.message);
        }

        // Fill form with test data
        const testInput = data.test_input;
        for (const key in testInput) {
            if (document.getElementById(key)) {
                document.getElementById(key).value = testInput[key];
            }
        }

        // Display results
        displayResults(data.test_result);
        alert('Model test successful! Model is working correctly.');
    } catch (error) {
        showError(error.message);
    } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

function loadSampleData() {
    // Sample high-value customer
    document.getElementById('Recency').value = 15;
    document.getElementById('Frequency').value = 12;
    document.getElementById('MonetaryValue').value = 1200;
    document.getElementById('Tenure').value = 730;
    document.getElementById('AvgDaysBetweenPurchases').value = 25;
    document.getElementById('Age').value = 42;
    document.getElementById('UniqueProductsCount').value = 5;
    document.getElementById('product_cost').value = 50.0;
}

function resetForm() {
    document.getElementById('customerForm').reset();
    document.getElementById('resultsContainer').classList.add('hidden');
    document.getElementById('errorContainer').classList.add('hidden');
    if (priceChart) {
        priceChart.destroy();
        priceChart = null;
    }
}

function displayResults(results) {
    // Update DOM with results
    document.getElementById('basePrice').textContent = results.base_price.toFixed(2);
    document.getElementById('dynamicPrice').textContent = results.dynamic_price.toFixed(2);
    document.getElementById('minPrice').textContent = results.min_price.toFixed(2);
    document.getElementById('profitMargin').textContent = results.profit_margin.toFixed(2);
    document.getElementById('clvValue').textContent = results.clv.toFixed(2);
    document.getElementById('adjustmentFactor').textContent = results.price_adjustment_factor.toFixed(2);

    // Show results container
    document.getElementById('resultsContainer').classList.remove('hidden');
    document.getElementById('errorContainer').classList.add('hidden');

    // Update chart
    updateChart(results);
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorContainer').classList.remove('hidden');
    document.getElementById('resultsContainer').classList.add('hidden');
}

function updateChart(results) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Base Price', 'Dynamic Price', 'Minimum Price'],
            datasets: [{
                label: 'Price Comparison',
                data: [results.base_price, results.dynamic_price, results.min_price],
                backgroundColor: [
                    'rgba(52, 152, 219, 0.7)',
                    'rgba(46, 204, 113, 0.7)',
                    'rgba(231, 76, 60, 0.7)'
                ],
                borderColor: [
                    'rgba(52, 152, 219, 1)',
                    'rgba(46, 204, 113, 1)',
                    'rgba(231, 76, 60, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}