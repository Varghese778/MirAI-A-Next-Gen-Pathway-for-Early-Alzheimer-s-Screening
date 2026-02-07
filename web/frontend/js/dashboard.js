/**
 * MirAI Dashboard Module
 */

document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    checkAuth();

    // Load user info
    loadUserInfo();

    // Load assessment history
    loadHistory();

    // Start assessment button
    document.getElementById('startBtn')?.addEventListener('click', startAssessment);
});

async function checkAuth() {
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    const valid = await window.API.auth.validate();
    if (!valid) {
        window.API.session.clear();
        window.location.href = 'index.html';
    }
}

function loadUserInfo() {
    const username = window.API.session.username || 'User';
    document.getElementById('userName').textContent = username;
    document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
}

async function loadHistory() {
    const historyList = document.getElementById('historyList');

    try {
        const result = await window.API.questionnaire.getHistory();

        if (result.assessments && result.assessments.length > 0) {
            historyList.innerHTML = result.assessments.map(assessment => {
                const date = new Date(assessment.completed_at).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric'
                });
                const category = assessment.risk_category || 'unknown';
                const score = assessment.risk_score || 0;

                return `
                    <div class="history-item" onclick="viewResult('${assessment.id}')">
                        <span class="history-date">${date}</span>
                        <span class="history-score ${category}">
                            ${score.toFixed(1)}% - ${category.charAt(0).toUpperCase() + category.slice(1)} Risk
                        </span>
                    </div>
                `;
            }).join('');
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

async function startAssessment() {
    const btn = document.getElementById('startBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const result = await window.API.questionnaire.startAssessment();
        console.log('Start assessment result:', result);
        if (result.id) {
            sessionStorage.setItem('mirai_assessment_id', result.id);
            window.location.href = 'questionnaire.html';
        } else {
            throw new Error('No assessment ID returned');
        }
    } catch (error) {
        alert('Failed to start assessment. Please try again.');
        console.error('Start assessment error:', error);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function viewResult(assessmentId) {
    sessionStorage.setItem('mirai_assessment_id', assessmentId);
    window.location.href = 'results.html';
}

async function logout() {
    await window.API.auth.logout();
    window.location.href = 'index.html';
}
