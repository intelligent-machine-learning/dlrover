// DLRover Dashboard JavaScript

// WebSocket connection management
class DashboardWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 5000;
        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.onConnected();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.onMessage(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.onDisconnected();
                // Reconnect after interval
                setTimeout(() => this.connect(), this.reconnectInterval);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            setTimeout(() => this.connect(), this.reconnectInterval);
        }
    }

    onConnected() {
        // Implementation should be overridden by Vue app
        window.dispatchEvent(new CustomEvent('websocket-connected'));
    }

    onDisconnected() {
        // Implementation should be overridden by Vue app
        window.dispatchEvent(new CustomEvent('websocket-disconnected'));
    }

    onMessage(data) {
        // Implementation should be overridden by Vue app
        window.dispatchEvent(new CustomEvent('websocket-message', { detail: data }));
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
}

// Utility functions
const DashboardUtils = {
    // Format bytes to human readable
    formatBytes: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Format duration
    formatDuration: function(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
    },

    // Get status color
    getStatusColor: function(status) {
        const colors = {
            'RUNNING': '#10b981',
            'SUCCEEDED': '#3b82f6',
            'FAILED': '#ef4444',
            'PENDING': '#f59e0b',
            'FINISHED': '#6b7280'
        };
        return colors[status] || '#6b7280';
    },

    // Check if node is critical
    isCriticalNode: function(node) {
        return node.critical === true;
    },

    // Escape HTML special characters to prevent XSS
    escapeHtml: function(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    },

    // Parse log lines and add syntax highlighting (basic)
    parseLogs: function(logText) {
        if (!logText) return '';

        return logText.split('\n').map(line => {
            const escaped = DashboardUtils.escapeHtml(line);
            // Basic log level highlighting
            if (line.includes('ERROR') || line.includes('FATAL')) {
                return `<span class="text-red-400">${escaped}</span>`;
            } else if (line.includes('WARN')) {
                return `<span class="text-yellow-400">${escaped}</span>`;
            } else if (line.includes('INFO')) {
                return `<span class="text-green-400">${escaped}</span>`;
            } else if (line.includes('DEBUG')) {
                return `<span class="text-gray-400">${escaped}</span>`;
            } else {
                return `<span class="text-gray-300">${escaped}</span>`;
            }
        }).join('\n');
    }
};

// Chart initialization functions
const ChartUtils = {
    // Initialize resource usage chart
    initResourceChart: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        // Placeholder for ECharts initialization
        // Actual implementation would be in Vue component
        return {
            update: function(data) {
                console.log('Update resource chart with:', data);
            },
            dispose: function() {
                console.log('Dispose resource chart');
            }
        };
    },

    // Initialize training speed chart
    initSpeedChart: function(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return null;

        // Placeholder for ECharts initialization
        return {
            update: function(data) {
                console.log('Update speed chart with:', data);
            },
            dispose: function() {
                console.log('Dispose speed chart');
            }
        };
    }
};

// Auto-refresh handler
class AutoRefresher {
    constructor(interval = 5000) {
        this.interval = interval;
        this.timer = null;
        this.isActive = false;
        this.callback = null;
    }

    start(callback) {
        if (this.isActive) return;

        this.callback = callback;
        this.isActive = true;
        this.timer = setInterval(callback, this.interval);
    }

    stop() {
        if (!this.isActive) return;

        this.isActive = false;
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }

    setInterval(interval) {
        this.interval = interval;
        if (this.isActive) {
            this.stop();
            this.start(this.callback);
        }
    }
}

// Export for use in Vue.js or other scripts
window.DashboardWebSocket = DashboardWebSocket;
window.DashboardUtils = DashboardUtils;
window.ChartUtils = ChartUtils;
window.AutoRefresher = AutoRefresher;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard JavaScript loaded');

    // Add any global event listeners here
    window.addEventListener('error', function(e) {
        console.error('Global error:', e.error);
    });
});