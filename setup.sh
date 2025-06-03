#!/bin/bash

# McKinsey Product Intelligence Platform - Quick Setup Script
# This script helps you set up the project quickly

echo "🚀 McKinsey Product Intelligence Platform - Setup"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi

echo "✅ pip found"

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created from template"
    echo "⚠️  Please edit .env file and add your Perplexity API key"
else
    echo "✅ .env file already exists"
fi

# Create .streamlit directory if it doesn't exist
if [ ! -d ".streamlit" ]; then
    mkdir .streamlit
    echo "📁 Created .streamlit directory"
fi

# Create config.toml in .streamlit directory
if [ ! -f .streamlit/config.toml ]; then
    cp config.toml .streamlit/config.toml
    echo "✅ Streamlit config file created"
fi

# Create secrets template
if [ ! -f .streamlit/secrets.toml.example ]; then
    cp secrets.toml.example .streamlit/secrets.toml.example
    echo "✅ Secrets template created"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your Perplexity API key"
echo "2. Run: streamlit run app.py (replace 'app.py' with your main file)"
echo "3. Upload sample_data.csv to test the application"
echo ""
echo "📚 For deployment instructions, see DEPLOYMENT.md"
echo "📖 For more information, see README.md"