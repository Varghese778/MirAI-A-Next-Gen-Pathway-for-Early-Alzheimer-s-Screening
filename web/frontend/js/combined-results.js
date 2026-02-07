/**
 * MirAI Combined Results Module
 * Displays integrated Stage-1 + Stage-2 risk assessment
 */

let assessmentId = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    // Get assessment ID
    assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        window.location.href = 'dashboard.html';
        return;
    }

    // Load user info
    loadUserInfo();

    // Load combined results
    await loadCombinedResults();
});

function loadUserInfo() {
    const username = window.API.session.username || 'User';
    document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
}

async function loadCombinedResults() {
    try {
        const result = await window.API.stage2.getCombined(assessmentId);
        console.log('Combined result:', result);

        // Update Stage-1 display
        updateStage1Display(result);

        // Update Stage-2 display
        updateStage2Display(result);

        // Update Combined display
        updateCombinedDisplay(result);

        // Update message
        document.getElementById('resultMessage').textContent = result.message;

    } catch (error) {
        console.error('Failed to load combined results:', error);
        document.getElementById('resultMessage').textContent =
            'Failed to load results. Please try again.';
    }
}

function updateStage1Display(result) {
    const score = result.stage1_risk_score || 0;
    const category = result.stage1_risk_category || 'unknown';

    document.getElementById('stage1Score').textContent = score.toFixed(1);
    document.getElementById('stage1GaugeFill').style.width = `${score}%`;

    const badge = document.getElementById('stage1Category');
    badge.textContent = capitalizeFirst(category) + ' Risk';
    badge.className = `risk-category-badge ${category}`;
}

function updateStage2Display(result) {
    const score = result.stage2_risk_score || 0;
    const category = result.stage2_risk_category || 'unknown';

    document.getElementById('stage2Score').textContent = score.toFixed(1);
    document.getElementById('stage2GaugeFill').style.width = `${score}%`;

    const badge = document.getElementById('stage2Category');
    badge.textContent = capitalizeFirst(category) + ' Risk';
    badge.className = `risk-category-badge ${category}`;
}

function updateCombinedDisplay(result) {
    const score = result.combined_risk_score || 0;
    const category = result.combined_risk_category || 'unknown';

    // Main gauge
    document.getElementById('combinedScore').textContent = score.toFixed(1);
    document.getElementById('combinedCategory').textContent = capitalizeFirst(category) + ' Combined Risk';
    document.getElementById('combinedCategory').className = `gauge-label ${category}`;

    // SVG gauge animation
    const gaugeFill = document.getElementById('combinedGaugeFill');
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (score / 100) * circumference;
    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = offset;

    // Set color based on category
    if (category === 'low') {
        gaugeFill.style.stroke = 'var(--success-color)';
    } else if (category === 'moderate') {
        gaugeFill.style.stroke = 'var(--warning-color)';
    } else {
        gaugeFill.style.stroke = 'var(--danger-color)';
    }

    // Small combined display
    document.getElementById('combinedScoreSmall').textContent = score.toFixed(1);

    const smallBadge = document.getElementById('combinedCategorySmall');
    smallBadge.textContent = capitalizeFirst(category);
    smallBadge.className = `risk-category-badge ${category}`;
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function goToStage3() {
    const assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (assessmentId) {
        window.location.href = 'stage3.html';
    } else {
        alert('No assessment found. Please start a new assessment.');
        window.location.href = 'dashboard.html';
    }
}
