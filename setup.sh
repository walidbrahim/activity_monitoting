#!/bin/bash

# Setup colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Room Activity Monitor Setup ===${NC}"

# 1. Check Python version
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}Error: python3 could not be found. Please install Python 3.9+.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$PYTHON_VERSION < 3.9" | bc -l) )); then
    echo -e "${RED}Error: Python version $PYTHON_VERSION is too old. Please use 3.9+.${NC}"
    exit 1
fi

echo -e "${GREEN}Python version $PYTHON_VERSION detected.${NC}"

# 2. Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# 3. Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
./venv/bin/pip install --upgrade pip

# 4. Install dependencies
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"
./venv/bin/pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Setup Complete! ===${NC}"
    echo -e "To start the application, run:"
    echo -e "${BLUE}source venv/bin/activate && python3 main.py${NC}"
else
    echo -e "${RED}Error: Failed to install dependencies.${NC}"
    exit 1
fi
