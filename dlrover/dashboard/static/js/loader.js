// Show a loading indicator while page resources are loading

(function() {
    // Create loading overlay
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.9); z-index: 9999; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <div class="animate-spin rounded-full h-32 w-32 border-b-2 border-purple-600 mb-4"></div>
            <p class="text-xl text-gray-700">Loading DLRover Dashboard...</p>
            <p class="text-sm text-gray-500 mt-2">Connecting to job monitoring service</p>
        </div>
    `;
    document.body.appendChild(overlay);

    // Remove loading overlay when page is ready
    function removeLoadingOverlay() {
        const loadingEl = document.getElementById('loading-overlay');
        if (loadingEl) {
            loadingEl.style.opacity = '0';
            setTimeout(() => loadingEl.remove(), 300);
        }
    }

    // Check if Vue.js and dependencies are loaded
    function checkDependencies() {
        if (typeof Vue !== 'undefined' &&
            document.readyState === 'complete') {
            removeLoadingOverlay();
        } else {
            setTimeout(checkDependencies, 100);
        }
    }

    // Start checking
    checkDependencies();

    // Fallback: remove after 5 seconds max
    setTimeout(removeLoadingOverlay, 5000);
})();