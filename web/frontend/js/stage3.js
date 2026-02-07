/**
 * MirAI Stage-3 Module
 * Biomarker assessment form handling
 */

let assessmentId = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    // Validate session with server
    try {
        const valid = await window.API.auth.validate();
        if (!valid) {
            alert('Your session has expired. Please log in again.');
            window.API.session.clear();
            window.location.href = 'index.html';
            return;
        }
    } catch (e) {
        console.error('Session validation error:', e);
        window.API.session.clear();
        window.location.href = 'index.html';
        return;
    }

    // Get assessment ID from session
    assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        alert('No assessment found. Please complete Stage-1 and Stage-2 first.');
        window.location.href = 'dashboard.html';
        return;
    }

    // Load user info
    loadUserInfo();

    // Form submission handler
    document.getElementById('stage3Form').addEventListener('submit', handleSubmit);

    // Prevent Enter key from submitting
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
            }
        });
    });
});

function loadUserInfo() {
    const username = window.API.session.username || 'User';
    document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
}

async function handleSubmit(e) {
    e.preventDefault();

    const btn = document.getElementById('submitBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        // Get form values (allow nulls for missing values)
        const ptau217Input = document.getElementById('ptau217').value;
        const ratioInput = document.getElementById('ptau217_abeta42_ratio').value;
        const nflInput = document.getElementById('nfl').value;

        const ptau217 = ptau217Input ? parseFloat(ptau217Input) : null;
        const ptau217_abeta42_ratio = ratioInput ? parseFloat(ratioInput) : null;
        const nfl = nflInput ? parseFloat(nflInput) : null;

        // Validate at least one biomarker is provided
        if (ptau217 === null && ptau217_abeta42_ratio === null && nfl === null) {
            alert('Please provide at least one biomarker value.');
            btn.classList.remove('loading');
            btn.disabled = false;
            return;
        }

        // Validate consent
        if (!document.getElementById('consentCheckbox').checked) {
            alert('Please acknowledge the medical disclaimer.');
            btn.classList.remove('loading');
            btn.disabled = false;
            return;
        }

        // Submit to API
        const result = await window.API.stage3.submit(assessmentId, {
            ptau217,
            ptau217_abeta42_ratio,
            nfl,
            consent_given: true
        });

        console.log('Stage-3 result:', result);

        // Store result and navigate to final results
        sessionStorage.setItem('mirai_stage3_complete', 'true');
        window.location.href = 'final-results.html';

    } catch (error) {
        console.error('Stage-3 submission error:', error);
        alert('Failed to submit biomarker data: ' + error.message);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}
