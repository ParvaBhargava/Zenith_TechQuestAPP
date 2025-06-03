# McKinsey Product Intelligence Platform

An AI-powered product analysis platform featuring comprehensive guardrails, executive summaries, and intelligent insights using Perplexity AI.

## 🚀 Features

- **Product Analysis**: Deep-dive analysis of individual products with customer review insights
- **Product Comparison**: Side-by-side intelligent comparison with actionable recommendations
- **Dataset Assistant**: Natural language Q&A about your entire product dataset
- **AI Guardrails**: Content safety features including profanity filtering, confidence detection, and hallucination risk assessment
- **Executive Summaries**: Concise insights before detailed analysis
- **Recent Review Priority**: Emphasis on recent customer feedback for more relevant insights

## 🛡️ AI Safety Features

### Content Guardrails
- **Profanity Detection**: Advanced ML-based inappropriate content filtering
- **Confidence Assessment**: Identifies uncertain or low-confidence responses
- **Hallucination Risk**: Detects unsupported claims and flags potential misinformation
- **Input/Output Validation**: Comprehensive safety checks for both prompts and responses

## 📋 Prerequisites

- Python 3.8 or higher
- Perplexity AI API key
- CSV dataset with product reviews

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd mckinsey-product-intelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   API_KEY=your_perplexity_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run your_main_file.py
   ```

## 📊 Dataset Format

Your CSV file should include these columns:
- `product_name`: Name of the product
- `category`: Product category
- `rating`: Customer rating (0-5)
- `review_content`: Customer review text
- `discounted_price`: Product price
- `rating_count`: Number of reviews

## 🌐 Deployment

### Streamlit Community Cloud
1. Push your code to GitHub
2. Connect your repo to [Streamlit Cloud](https://share.streamlit.io/)
3. Add your `API_KEY` in the Streamlit Cloud secrets management
4. Deploy!

### Self-Hosting
The app can be deployed on:
- **Google Cloud Run**: ~$1-2/day for minimal traffic
- **AWS ECS/EC2**: Variable pricing based on usage
- **Azure Container Instances**: Pay-per-use model
- **Heroku**: Free tier available with limitations

## 🔑 API Configuration

1. **Get Perplexity API Key**
   - Sign up at [Perplexity AI](https://perplexity.ai/)
   - Generate API key from dashboard
   - Pricing: ~$0.20-$5 per 1M tokens

2. **Configure in App**
   - Enter API key in sidebar when running locally
   - For deployment, add to Streamlit Cloud secrets or environment variables

## 💰 Cost Optimization

- **Local Guardrails**: All safety checks run locally (no API costs)
- **Single API Call**: Optimized to minimize Perplexity API usage
- **Efficient Prompting**: Smart prompt engineering reduces token usage
- **Error Prevention**: Early validation prevents costly failed requests

## 🎯 Usage Examples

### Product Analysis
```
Select any product → Generate Analysis
Get executive summary + detailed insights
```

### Product Comparison
```
Choose 2 products → Generate Comparison
Receive head-to-head analysis with recommendations
```

### Dataset Assistant
```
Ask: "Which products offer best value under ₹1000?"
Get: Intelligent analysis with specific recommendations
```

## 🔒 Security & Privacy

- **No Data Storage**: Your data stays local during processing
- **API Security**: Secure HTTPS communication with Perplexity AI
- **Content Filtering**: Multiple layers of safety checks
- **Environment Variables**: Secure API key management

## 🤖 AI Models Used

| Model | Purpose | Strength | Cost Range |
|-------|---------|----------|------------|
| Perplexity Sonar Large | Primary Analysis | Quality & Context | $0.50-$1/1M tokens |
| Local ML Models | Guardrails | Safety & Speed | Free (local) |

## 📈 Future Enhancements

- **Agentic AI**: Autonomous monitoring and proactive insights
- **Multimodal Analysis**: Image and video review processing
- **Real-time Integration**: Live social media sentiment tracking
- **Advanced Visualization**: Interactive charts and dashboards

## 🐛 Troubleshooting

### Common Issues
1. **API Key Error**: Ensure valid Perplexity API key is configured
2. **CSV Format**: Check your dataset has required columns
3. **Dependencies**: Run `pip install -r requirements.txt` if modules missing
4. **Guardrails Warning**: `profanity-check` is optional - app works without it

### Support
- Create GitHub issue for bugs
- Check Streamlit documentation for deployment issues
- Verify API quotas if getting rate limits

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Contact

For questions or support, please create an issue in this repository.

---

**Built with ❤️ for McKinsey TechQuest 2025**