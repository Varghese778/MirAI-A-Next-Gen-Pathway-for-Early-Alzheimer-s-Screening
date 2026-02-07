/**
 * MirAI Results Module
 * Displays assessment results with visualizations
 */

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    const assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        window.location.href = 'dashboard.html';
        return;
    }

    await loadResults(assessmentId);
});

async function loadResults(assessmentId) {
    try {
        const result = await window.API.questionnaire.getResult(assessmentId);
        displayResults(result);
    } catch (error) {
        console.error('Failed to load results:', error);
        document.getElementById('messageText').textContent =
            'Failed to load results. Please return to the dashboard and try again.';
    }
}

function displayResults(data) {
    const riskScore = data.risk_score || 0;
    const riskCategory = data.risk_category || 'unknown';
    const factors = data.contributing_factors || [];

    // Update score display
    document.getElementById('riskScore').textContent = `${riskScore.toFixed(1)}%`;
    document.getElementById('riskScore').className = `risk-score-value ${riskCategory}`;

    // Update category
    const categoryEl = document.getElementById('riskCategory');
    categoryEl.textContent = `${riskCategory.charAt(0).toUpperCase() + riskCategory.slice(1)} Risk`;
    categoryEl.className = `risk-category ${riskCategory}`;

    // Animate gauge
    animateGauge(riskScore / 100, riskCategory);

    // Update message based on category
    const messages = {
        low: 'Your screening suggests a lower risk profile. Continue maintaining healthy lifestyle habits. Regular check-ups with your healthcare provider are still recommended, especially if you have family history of cognitive conditions.',
        moderate: 'Your screening indicates some risk factors that warrant attention. Consider discussing these results with a healthcare professional. Lifestyle modifications such as increased physical activity, cognitive engagement, and cardiovascular health management may be beneficial.',
        elevated: 'Your screening shows elevated risk indicators. We strongly recommend consulting with a healthcare professional for a comprehensive clinical evaluation. Early intervention and lifestyle modifications can make a significant difference in cognitive health outcomes.'
    };

    document.getElementById('messageText').textContent = messages[riskCategory] || messages.moderate;

    // Display contributing factors
    if (factors.length > 0) {
        const factorsList = document.getElementById('factorsList');
        factorsList.innerHTML = factors.map(factor => `
            <div class="factor-item">
                <span class="factor-label">${factor.label}</span>
                <span class="factor-value">+${(factor.contribution * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    } else {
        document.getElementById('factorsSection').style.display = 'none';
    }
}

function animateGauge(probability, category) {
    const gauge = document.getElementById('gaugeFill');
    gauge.className = `gauge-fill ${category}`;

    // SVG arc calculation
    const radius = 70;
    const circumference = Math.PI * radius;
    const dashOffset = circumference * (1 - probability);

    gauge.style.strokeDasharray = `${circumference}`;
    gauge.style.strokeDashoffset = `${circumference}`;

    // Animate after a short delay
    setTimeout(() => {
        gauge.style.strokeDashoffset = `${dashOffset}`;
    }, 100);
}

async function startNewAssessment() {
    try {
        const result = await window.API.questionnaire.startAssessment();
        if (result.id) {
            sessionStorage.setItem('mirai_assessment_id', result.id);
            window.location.href = 'questionnaire.html';
        }
    } catch (error) {
        console.error('Failed to start new assessment:', error);
        alert('Failed to start new assessment. Please try again.');
    }
}

function goToStage2() {
    const assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (assessmentId) {
        window.location.href = 'stage2.html';
    } else {
        alert('No assessment found. Please start a new assessment.');
        window.location.href = 'dashboard.html';
    }
}
