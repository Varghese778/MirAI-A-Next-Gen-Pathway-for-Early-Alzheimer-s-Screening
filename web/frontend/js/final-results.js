/**
 * MirAI Final Results Module
 * Displays combined risk from all three stages
 */

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    // Get assessment ID
    const assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        alert('No assessment found.');
        window.location.href = 'dashboard.html';
        return;
    }

    // Load user info
    loadUserInfo();

    // Load final results
    await loadFinalResults(assessmentId);
});

function loadUserInfo() {
    const username = window.API.session.username || 'User';
    document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
}

async function loadFinalResults(assessmentId) {
    try {
        const result = await window.API.stage3.getFinal(assessmentId);
        console.log('Final results:', result);

        // Update Stage-1
        updateStageDisplay('stage1', result.stage1_risk_score, result.stage1_risk_category);

        // Update Stage-2
        updateStageDisplay('stage2', result.stage2_risk_score, result.stage2_risk_category);

        // Update Stage-3
        updateStageDisplay('stage3', result.stage3_risk_score, result.stage3_risk_category);

        // Update Final
        updateFinalDisplay(result);

        // Update Recommendation
        updateRecommendation(result.final_risk_category, result.final_recommendation);

    } catch (error) {
        console.error('Failed to load final results:', error);
        document.getElementById('finalCategory').textContent = 'Error loading results';
        document.getElementById('recText').textContent = 'Failed to load recommendation. Please try again.';
    }
}

function updateStageDisplay(stage, score, category) {
    const scoreEl = document.getElementById(`${stage}Score`);
    const fillEl = document.getElementById(`${stage}Fill`);
    const catEl = document.getElementById(`${stage}Category`);

    if (scoreEl) scoreEl.textContent = score.toFixed(1);
    if (fillEl) fillEl.style.width = `${score}%`;
    if (catEl) {
        catEl.textContent = capitalizeFirst(category);
        catEl.className = `stage-category ${category}`;
    }

    // Set fill color
    if (fillEl) {
        if (category === 'low') {
            fillEl.style.background = 'var(--success)';
        } else if (category === 'moderate') {
            fillEl.style.background = 'var(--warning)';
        } else {
            fillEl.style.background = 'var(--danger)';
        }
    }
}

function updateFinalDisplay(result) {
    const score = result.final_risk_score;
    const category = result.final_risk_category;

    // Update score
    document.getElementById('finalScore').textContent = score.toFixed(1);

    // Update category label
    const catEl = document.getElementById('finalCategory');
    catEl.textContent = `${capitalizeFirst(category)} Final Risk`;
    catEl.className = `gauge-label ${category}`;

    // Animate gauge
    const gaugeFill = document.getElementById('finalGaugeFill');
    const circumference = 2 * Math.PI * 54;
    const offset = circumference - (score / 100) * circumference;
    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = offset;

    // Set color
    if (category === 'low') {
        gaugeFill.style.stroke = 'var(--success)';
    } else if (category === 'moderate') {
        gaugeFill.style.stroke = 'var(--warning)';
    } else {
        gaugeFill.style.stroke = 'var(--danger)';
    }
}

function updateRecommendation(category, recommendation) {
    const card = document.getElementById('recommendationCard');
    const icon = document.getElementById('recIcon');
    const text = document.getElementById('recText');

    text.textContent = recommendation;

    // Style based on category
    if (category === 'low') {
        icon.textContent = '✅';
        card.classList.add('rec-low');
    } else if (category === 'moderate') {
        icon.textContent = '⚠️';
        card.classList.add('rec-moderate');
    } else {
        icon.textContent = '🔴';
        card.classList.add('rec-high');
    }
}

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}
