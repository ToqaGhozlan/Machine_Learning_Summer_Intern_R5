document.addEventListener('DOMContentLoaded', function() {
    
    // Fill Default Data Button
    const fillBtn = document.getElementById('fillDefaultBtn');
    if (fillBtn && Object.keys(defaultData).length > 0) {
        fillBtn.addEventListener('click', function() {
            
            // Map the JSON keys to the HTML element IDs
            const fields = [
                'temp_mean',
                'humidity',
                'wind_speed',
                'precipitation'
            ];
            
            fields.forEach(field => {
                const el = document.getElementById(field);
                if(el && defaultData[field] !== undefined) {
                    el.value = defaultData[field];
                    
                    // Flash valid style for visual feedback
                    el.classList.add('is-valid');
                    setTimeout(() => el.classList.remove('is-valid'), 1000);
                }
            });
        });
    }

});
