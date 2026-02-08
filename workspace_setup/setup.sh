#!/bin/bash
#
# AetherNav Stack Setup Script
# Creates virtual environment and installs all dependencies
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print stylish banner
print_banner() {
    echo -e "${CYAN}"
    echo "    ╔═══════════════════════════════════════════════════════════════╗"
    echo "    ║                                                               ║"
    echo -e "    ║     ${MAGENTA}█████╗ ███████╗████████╗██╗  ██╗███████╗██████╗ ${CYAN}          ║"
    echo -e "    ║    ${MAGENTA}██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗${CYAN}         ║"
    echo -e "    ║    ${MAGENTA}███████║█████╗     ██║   ███████║█████╗  ██████╔╝${CYAN}         ║"
    echo -e "    ║    ${MAGENTA}██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗${CYAN}         ║"
    echo -e "    ║    ${MAGENTA}██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║${CYAN}         ║"
    echo -e "    ║    ${MAGENTA}╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝${CYAN}         ║"
    echo "    ║                                                               ║"
    echo -e "    ║           ${YELLOW}███╗   ██╗ █████╗ ██╗   ██╗${CYAN}                        ║"
    echo -e "    ║           ${YELLOW}████╗  ██║██╔══██╗██║   ██║${CYAN}                        ║"
    echo -e "    ║           ${YELLOW}██╔██╗ ██║███████║██║   ██║${CYAN}                        ║"
    echo -e "    ║           ${YELLOW}██║╚██╗██║██╔══██║╚██╗ ██╔╝${CYAN}                        ║"
    echo -e "    ║           ${YELLOW}██║ ╚████║██║  ██║ ╚████╔╝ ${CYAN}                        ║"
    echo -e "    ║           ${YELLOW}╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ${CYAN}                        ║"
    echo "    ║                                                               ║"
    echo -e "    ║        ${GREEN}Autonomous Vehicle Navigation Stack${CYAN}                   ║"
    echo "    ║                                                               ║"
    echo "    ╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print section header
print_section() {
    echo -e "\n${BOLD}${GREEN}▶ $1${NC}\n"
}

# Print info message
print_info() {
    echo -e "  ${CYAN}ℹ${NC} $1"
}

# Print success message
print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

# Print warning message
print_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

# Print error message
print_error() {
    echo -e "  ${RED}✗${NC} $1"
}

# Ask yes/no question
ask_yes_no() {
    while true; do
        read -p "  $1 [y/n]: " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "  Please answer y or n.";;
        esac
    done
}

# Main setup
main() {
    print_banner
    
    echo -e "${BOLD}Starting AetherNav Setup...${NC}"
    echo -e "Project root: ${CYAN}$PROJECT_ROOT${NC}\n"
    
    # =========================================
    # Git: Pull latest code
    # =========================================
    print_section "Git Repository"
    
    if [ -d "$PROJECT_ROOT/.git" ]; then
        print_info "Git repository detected"
        
        # Show current branch
        CURRENT_BRANCH=$(cd "$PROJECT_ROOT" && git branch --show-current)
        print_info "Current branch: ${YELLOW}$CURRENT_BRANCH${NC}"
        
        if ask_yes_no "Pull latest code from ${YELLOW}dev/refactoring${NC} branch?"; then
            print_info "Fetching from remote..."
            cd "$PROJECT_ROOT"
            git fetch origin
            
            if git checkout dev/refactoring 2>/dev/null; then
                print_success "Switched to dev/refactoring branch"
            else
                print_warning "Branch dev/refactoring not found, staying on $CURRENT_BRANCH"
            fi
            
            git pull origin "$(git branch --show-current)" || print_warning "Pull failed, continuing..."
            print_success "Repository updated"
        else
            print_info "Skipping git pull"
        fi
    else
        print_warning "Not a git repository, skipping git operations"
    fi
    
    # =========================================
    # Virtual Environment
    # =========================================
    print_section "Virtual Environment"
    
    VENV_DIR="$PROJECT_ROOT/.aethernav_env"
    
    if [ -d "$VENV_DIR" ]; then
        print_info "Virtual environment already exists at: $VENV_DIR"
        if ask_yes_no "Remove and recreate virtual environment?"; then
            print_info "Removing existing environment..."
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created: $VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
    print_info "Python: $(which python)"
    print_info "Version: $(python --version)"
    
    # =========================================
    # Install Dependencies
    # =========================================
    print_section "Installing Dependencies"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip -q
    
    # Install requirements
    REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
    
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_info "Installing from requirements.txt..."
        pip install -r "$REQUIREMENTS_FILE"
        print_success "Core dependencies installed"
    else
        print_error "requirements.txt not found at: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # =========================================
    # Verify Installation
    # =========================================
    print_section "Verifying Installation"
    
    # Test imports
    python -c "
import numpy
import cv2
import yaml
import onnxruntime
import matplotlib
print('All core imports successful')
" && print_success "All dependencies verified" || print_error "Some imports failed"
    
    # =========================================
    # Summary
    # =========================================
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}  Setup Complete!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  To activate the environment:"
    echo -e "    ${YELLOW}source $VENV_DIR/bin/activate${NC}"
    echo ""
    echo -e "  To run AetherNav:"
    echo -e "    ${YELLOW}cd $PROJECT_ROOT${NC}"
    echo -e "    ${YELLOW}python aethernav_stack/run.py${NC}"
    echo ""
    echo -e "  To run trajectory test:"
    echo -e "    ${YELLOW}python -m aethernav_stack.tests.run_trajectory_test --duration 60${NC}"
    echo ""
}

# Run main
main
