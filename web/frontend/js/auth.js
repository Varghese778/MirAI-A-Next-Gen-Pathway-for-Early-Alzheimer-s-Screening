/**
 * MirAI Authentication Module
 * Handles login, registration, and session management
 */

document.addEventListener('DOMContentLoaded', () => {
    // Check if user is already logged in
    checkSession();

    // Form elements
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    const showRegister = document.getElementById('showRegister');
    const showLogin = document.getElementById('showLogin');
    const authError = document.getElementById('authError');

    // Toggle between login and register
    showRegister?.addEventListener('click', (e) => {
        e.preventDefault();
        loginForm.classList.remove('active');
        registerForm.classList.add('active');
        hideError();
    });

    showLogin?.addEventListener('click', (e) => {
        e.preventDefault();
        registerForm.classList.remove('active');
        loginForm.classList.add('active');
        hideError();
    });

    // Login form submission
    loginForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = loginForm.querySelector('button[type="submit"]');
        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;

        if (!username || !password) {
            showError('Please fill in all fields');
            return;
        }

        setLoading(btn, true);
        hideError();

        try {
            const result = await window.API.auth.login(username, password);
            if (result.success) {
                window.location.href = 'dashboard.html';
            }
        } catch (error) {
            showError(error.message || 'Login failed. Please try again.');
        } finally {
            setLoading(btn, false);
        }
    });

    // Register form submission
    registerForm?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = registerForm.querySelector('button[type="submit"]');
        const username = document.getElementById('registerUsername').value.trim();
        const password = document.getElementById('registerPassword').value;
        const confirm = document.getElementById('confirmPassword').value;

        if (!username || !password || !confirm) {
            showError('Please fill in all fields');
            return;
        }

        if (username.length < 3) {
            showError('Username must be at least 3 characters');
            return;
        }

        if (password.length < 6) {
            showError('Password must be at least 6 characters');
            return;
        }

        if (password !== confirm) {
            showError('Passwords do not match');
            return;
        }

        setLoading(btn, true);
        hideError();

        try {
            const result = await window.API.auth.register(username, password);
            if (result.success) {
                window.location.href = 'dashboard.html';
            }
        } catch (error) {
            showError(error.message || 'Registration failed. Please try again.');
        } finally {
            setLoading(btn, false);
        }
    });

    // Helper functions
    function showError(message) {
        if (authError) {
            authError.querySelector('.error-text').textContent = message;
            authError.classList.remove('hidden');
        }
    }

    function hideError() {
        if (authError) {
            authError.classList.add('hidden');
        }
    }

    function setLoading(btn, loading) {
        if (loading) {
            btn.classList.add('loading');
            btn.disabled = true;
        } else {
            btn.classList.remove('loading');
            btn.disabled = false;
        }
    }

    async function checkSession() {
        if (window.API.session.isValid()) {
            const valid = await window.API.auth.validate();
            if (valid) {
                window.location.href = 'dashboard.html';
            } else {
                window.API.session.clear();
            }
        }
    }
});

// Logout function
async function logout() {
    await window.API.auth.logout();
    window.location.href = 'index.html';
}
