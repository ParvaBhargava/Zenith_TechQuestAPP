# Deployment Guide for McKinsey Product Intelligence Platform

## 📋 Pre-Deployment Checklist

- [ ] All code files are ready and tested locally
- [ ] `requirements.txt` includes all dependencies
- [ ] `.gitignore` excludes sensitive files
- [ ] `README.md` provides clear instructions
- [ ] API key is obtained from Perplexity AI
- [ ] Sample data is available for testing

## 🌐 Streamlit Community Cloud Deployment (Recommended)

### Step 1: Prepare GitHub Repository
1. Create a new GitHub repository
2. Upload all your files:
   - Your main Python file (e.g., `app.py` or `main.py`)
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `config.toml` (optional)
   - `sample_data.csv` (for testing)

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Choose your main Python file
6. Click "Deploy"

### Step 3: Configure Secrets
1. In Streamlit Cloud dashboard, click on your app
2. Go to "Settings" → "Secrets"
3. Add your API key:
   ```toml
   API_KEY = "your_perplexity_api_key_here"
   ```
4. Save secrets

### Step 4: Test Deployment
1. Wait for deployment to complete
2. Access your app via the provided URL
3. Test with the sample data file
4. Verify all features work correctly

## 🔧 Self-Hosting Options

### Option 1: Google Cloud Run
```bash
# Build Docker image
docker build -t mckinsey-product-intelligence .

# Deploy to Cloud Run
gcloud run deploy mckinsey-product-intelligence \
  --image gcr.io/YOUR_PROJECT/mckinsey-product-intelligence \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 2: Heroku
```bash
# Install Heroku CLI and login
heroku login

# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set API_KEY=your_perplexity_api_key_here

# Deploy
git push heroku main
```

### Option 3: AWS EC2
1. Launch EC2 instance with Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run: `streamlit run your_main_file.py --server.port 8501`
5. Configure security groups for port 8501

## 🔑 Environment Configuration

### Local Development
1. Copy `.env.example` to `.env`
2. Add your Perplexity API key
3. Run: `streamlit run your_main_file.py`

### Production Deployment
- **Streamlit Cloud**: Use Secrets management
- **Heroku**: Use Config Vars
- **AWS/GCP**: Use environment variables or secret managers
- **Docker**: Use environment variables or Docker secrets

## 🚨 Common Deployment Issues

### Issue: Module Not Found
**Solution**: Ensure all dependencies are in `requirements.txt`

### Issue: API Key Not Working
**Solution**: 
- Verify API key is correctly set in secrets/environment
- Check API key permissions and quotas
- Ensure no extra spaces or characters

### Issue: File Upload Fails
**Solution**: 
- Check Streamlit file upload size limits
- Verify CSV format matches expected schema
- Ensure file permissions are correct

### Issue: Slow Performance
**Solution**:
- Use smaller sample data for testing
- Optimize API calls and caching
- Consider upgrading hosting plan

## 📊 Monitoring & Maintenance

### Health Checks
- Test all three main features regularly
- Monitor API usage and costs
- Check error logs for issues

### Updates
- Keep dependencies updated
- Monitor Perplexity API changes
- Update guardrails as needed

### Scaling
- Monitor user traffic
- Upgrade hosting resources if needed
- Consider caching for high-traffic scenarios

## 💰 Cost Estimation

### Streamlit Community Cloud
- **Free tier**: 1 private app + unlimited public apps
- **Cost**: $0/month for most use cases

### Self-Hosting Costs
- **Google Cloud Run**: ~$1-2/day for minimal traffic
- **Heroku**: $7/month (Hobby tier) + $25/month (Basic tier)
- **AWS EC2**: $5-20/month depending on instance size

### API Costs
- **Perplexity API**: ~$0.20-$5 per 1M tokens
- **Typical usage**: $1-10/month for moderate use

## 🔐 Security Best Practices

1. **Never commit API keys to Git**
2. **Use HTTPS for all deployments**
3. **Regularly rotate API keys**
4. **Monitor API usage for anomalies**
5. **Keep dependencies updated**
6. **Use strong passwords for hosting accounts**

## 📞 Support Resources

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Perplexity API**: [docs.perplexity.ai](https://docs.perplexity.ai)
- **GitHub Issues**: Create issues in your repository
- **Community**: Streamlit Community Forum

---

**Ready to deploy? Follow the Streamlit Cloud steps above for the easiest deployment experience!**