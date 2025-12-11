#!/bin/bash

# =============================================================================
# Delivery ETA Prediction System - Run Script (Linux/macOS)
# =============================================================================
# This script sets up and runs all components of the project
# Usage: bash run.sh [command]
# Commands: setup, train, api, dashboard, all
# =============================================================================

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}  🚚 Delivery ETA Prediction System${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment
activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        print_step "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        print_warning "Virtual environment not found. Creating one..."
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
    fi
}

# Install dependencies
install_deps() {
    print_step "Installing dependencies..."
    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
}

# Setup: Create venv and install dependencies
setup() {
    print_header
    echo ""
    print_step "Setting up the project..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_step "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    activate_venv
    install_deps
    
    echo ""
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run training: bash run.sh train"
    echo "  2. Start API: bash run.sh api"
    echo "  3. Start Dashboard: bash run.sh dashboard"
}

# Train the model
train() {
    print_header
    echo ""
    print_step "Training the model..."
    
    activate_venv
    cd "$PROJECT_DIR"
    python src/models/train_model.py
    
    echo ""
    echo -e "${GREEN}✅ Training complete! Model saved to models/best_model.pkl${NC}"
}

# Start FastAPI server
start_api() {
    print_header
    echo ""
    print_step "Starting FastAPI server..."
    
    activate_venv
    cd "$PROJECT_DIR"
    
    echo ""
    echo "API Documentation: http://localhost:8000/docs"
    echo "Health Check: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
}

# Start Streamlit dashboard
start_dashboard() {
    print_header
    echo ""
    print_step "Starting Streamlit dashboard..."
    
    activate_venv
    cd "$PROJECT_DIR"
    
    echo ""
    echo "Dashboard: http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop the dashboard"
    echo ""
    
    streamlit run dashboard/app.py --server.port 8501
}

# Run all components (train + start both servers)
run_all() {
    print_header
    echo ""
    print_step "Running complete pipeline..."
    
    activate_venv
    cd "$PROJECT_DIR"
    
    # Train first
    print_step "Step 1/3: Training model..."
    python src/models/train_model.py
    
    echo ""
    echo -e "${GREEN}✅ Training complete!${NC}"
    echo ""
    
    # Start servers in background
    print_step "Step 2/3: Starting API server (background)..."
    uvicorn api.app:app --host 0.0.0.0 --port 8000 &
    API_PID=$!
    sleep 3
    
    print_step "Step 3/3: Starting dashboard..."
    echo ""
    echo "API running at: http://localhost:8000/docs"
    echo "Dashboard running at: http://localhost:8501"
    echo ""
    echo "Press Ctrl+C to stop all services"
    
    # Handle cleanup on exit
    trap "kill $API_PID 2>/dev/null; exit" SIGINT SIGTERM
    
    streamlit run dashboard/app.py --server.port 8501
}

# Show help
show_help() {
    print_header
    echo ""
    echo "Usage: bash run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup      - Create virtual environment and install dependencies"
    echo "  train      - Train the ML model"
    echo "  api        - Start the FastAPI prediction server"
    echo "  dashboard  - Start the Streamlit dashboard"
    echo "  all        - Train model and start both servers"
    echo "  help       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  bash run.sh setup     # First time setup"
    echo "  bash run.sh train     # Train the model"
    echo "  bash run.sh api       # Start API only"
    echo "  bash run.sh dashboard # Start dashboard only"
    echo "  bash run.sh all       # Run everything"
    echo ""
}

# Main command handler
case "${1:-help}" in
    setup)
        setup
        ;;
    train)
        train
        ;;
    api)
        start_api
        ;;
    dashboard)
        start_dashboard
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
