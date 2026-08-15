document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');

    if (form && submitBtn) {
        form.addEventListener('submit', function() {
            submitBtn.classList.add('loading');
            submitBtn.innerHTML = `
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
                Processing...
            `;
        });
    }

    const inputs = document.querySelectorAll('.form-input, .form-select');
    inputs.forEach(function(input) {
        input.addEventListener('focus', function() {
            this.closest('.field-group').classList.add('field-focused');
        });
        input.addEventListener('blur', function() {
            this.closest('.field-group').classList.remove('field-focused');
        });
    });

    const resultPanel = document.getElementById('result-panel');
    if (resultPanel && resultPanel.classList.contains('result-visible')) {
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
