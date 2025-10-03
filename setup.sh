#!/bin/bash

# Crop Disease Detection - Setup Script
# This script automates the initial project setup

echo "============================================"
echo "🌱 Crop Disease Detection - Setup"
echo "============================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}1. Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo -e "\n${YELLOW}2. Creating virtual environment...${NC}"
python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo -e "\n${YELLOW}3. Activating virtual environment...${NC}"
source venv/bin/activate || . venv/Scripts/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${YELLOW}4. Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install requirements
echo -e "\n${YELLOW}5. Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directory structure
echo -e "\n${YELLOW}6. Creating directory structure...${NC}"
mkdir -p data/raw
mkdir -p data/processed/{train,val,test}
mkdir -p models/saved_models
mkdir -p results
mkdir -p logs
mkdir -p notebooks
mkdir -p app/static
mkdir -p examples

echo -e "${GREEN}✓ Directory structure created${NC}"

# Create empty files
touch data/processed/class_names.json
touch results/.gitkeep
touch logs/.gitkeep

# Display next steps
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"

echo -e "\n📋 Next Steps:"
echo -e "   1. Download PlantVillage dataset:"
echo -e "      ${YELLOW}kaggle datasets download -d abdallahalidev/plantvillage-dataset${NC}"
echo -e "      ${YELLOW}unzip plantvillage-dataset.zip -d data/raw/${NC}"
echo -e ""
echo -e "   2. Run Exploratory Data Analysis:"
echo -e "      ${YELLOW}jupyter notebook notebooks/01_eda.ipynb${NC}"
echo -e ""
echo -e "   3. Prepare data (train/val/test split):"
echo -e "      ${YELLOW}python src/data_loader.py${NC}"
echo -e ""
echo -e "   4. Train the model:"
echo -e "      ${YELLOW}python src/train.py --model mobilenet_v2 --epochs 30${NC}"
echo -e ""
echo -e "   5. Run the web application:"
echo -e "      ${YELLOW}streamlit run app/app.py${NC}"
echo -e ""
echo -e "💡 Tips:"
echo -e "   - Use GPU for faster training (CUDA required)"
echo -e "   - Start with MobileNetV2 for best speed/accuracy tradeoff"
echo -e "   - Monitor training with TensorBoard: ${YELLOW}tensorboard --logdir logs/${NC}"
echo -e ""
echo -e "📚 Documentation: See README.md for detailed instructions"
echo -e ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
echo -e ""