from dlrover.python.master.node.job_context import get_job_context

# DLRover Dashboard

A modern, real-time visualization dashboard for DLRover distributed training jobs on Kubernetes.

## Features

- 🎯 **Job Overview**: Real-time job status, stage tracking, and node statistics
- 🖥️ **Node Visualization**: Interactive display of node states (Worker, PS, Chief, Evaluator)
- 📊 **Resource Monitoring**: CPU, memory, and GPU usage across all nodes
- 🛡️ **Fault Tolerance**: View restarts, failures, and exit reasons
- 📝 **Log Streaming**: Real-time log viewing for individual nodes
- ⚡ **Real-time Updates**: WebSocket-based live updates every 2 seconds
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices

## Setup

### Prerequisites

- Python 3.7+
- A running DLRover master (for integration mode)

### Installation

Install required dependencies:
```bash
pip install dlrover[k8s]
```

## Usage

### Standalone Mode

Run the dashboard as a standalone service:

```bash
python -m dlrover.dashboard.run_dashboard --host 0.0.0.0 --port 8080
```

Options:
- `--host`: Host address (default: 0.0.0.0)
- `--port`: Port number (default: 8080)
- `--log-dir`: Directory containing log files (default: /tmp/dlrover)

### Integration Mode

Integrate the dashboard with DLRover Master:

```python
from dlrover.python.master.dist_master import DistributedMaster
from dlrover.dashboard.integrate_with_master import add_dashboard_to_master

# Create master server
master = DistributedMaster(...)

# Add dashboard with custom config
dashboard_config = {
    "enable": True,
    "host": "0.0.0.0",
    "port": 8080
}
add_dashboard_to_master(master, dashboard_config)

# Start master (dashboard will automatically start)
master.start()
```

## Architecture

### Backend Components

1. **Dashboard Application (`app.py`)**:
   - Tornado-based web server
   - REST API endpoints for job, node, and metrics data
   - WebSocket support for real-time updates
   - Static file serving for UI

2. **Service Integration (`service_integration.py`)**:
   - DashboardService for metrics collection
   - Global step tracking
   - Training speed calculation
   - Session information aggregation

3. **Master Integration (`integrate_with_master.py`)**:
   - DashboardManager for lifecycle management
   - Broadcast loop for real-time updates
   - Seamless integration with master server

### Frontend Features

- **Vue.js 3**: Reactive framework for dynamic UI
- **Tailwind CSS**: Utility-first CSS for modern styling
- **ECharts**: Data visualization library (pre-loaded for future charts)
- **WebSockets**: Real-time bidirectional communication
- **Responsive Design**: Mobile-friendly layout

### API Endpoints

- `GET /api/job` - Overall job information
- `GET /api/nodes` - All node details
- `GET /api/logs/<node_name>` - Node logs
- `GET /api/diagnosis` - Diagnosis results and fault tolerance info
- `GET /api/metrics` - Real-time metrics (CPU, memory, GPU, training speed)
- `WebSocket /ws` - Real-time updates

### Data Sources

The dashboard integrates with existing DLRover components:
- **JobContext**: Node states, job stages, fault information
- **DiagnosisDataManager**: Health checks and diagnosis results
- **PerformanceMonitor**: Training metrics and global step tracking
- **Log files**: Node-specific log files from distributed workers

## Configuration

### Environment Variables

- `DLROVER_LOG_DIR`: Directory containing node log files (default: /tmp/dlrover)

### Log File Convention

Log files should follow the naming convention:
- `<node_name>.log` - e.g., `dlrover-job-123-worker-1.log`

## Development

### Extending the Dashboard

1. **Add new API endpoints** in `app.py`
2. **Update Vue.js components** in `templates/index.html`
3. **Add CSS styles** using Tailwind utilities

### Mock Data for Testing

User can use mock dashboard directly:

```text
http://0.0.0.0:8080?mock=true
```

During development, you can use mock data:

```python
# Add to app.py for testing
MOCK_DATA = True

if MOCK_DATA and not get_job_context():
    # Return mock data for development
    self.write(json.dumps({"mock": "data"}))
    return
```

## Troubleshooting

### Dashboard not displaying data

1. Check if DLRover master is running
2. Verify JobContext has job data
3. Enable debug logging: `--log debug`

### WebSocket connection issues

1. Check firewall settings
2. Ensure reverse proxy (if any) supports WebSocket
3. Verify browser supports WebSocket

### Log viewing issues

1. Verify log file location: `DLROVER_LOG_DIR`
2. Check file permissions
3. Verify node name matching

## Future Enhancements

- [ ] Resource usage charts (CPU, Memory, GPU over time)
- [ ] Training progress visualization
- [ ] Node topology diagram
- [ ] Click-to-scale functionality
- [ ] Training metric graphs (loss, accuracy)
- [ ] Alert system for failures
- [ ] Export data functionality
- [ ] Kubernetes event integration

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

Apache 2.0 - See LICENSE file for details