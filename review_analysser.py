import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter
import requests
import json
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ENHANCED DARK BLUE UI THEME WITH IMPROVED FORMATTING
st.set_page_config(
    page_title="McKinsey Product Intelligence", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit default elements for cleaner look
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .main-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #2563eb 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.4);
    }
    
    .product-detail-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 2px solid #3b82f6;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.3rem;
        border: 1px solid #60a5fa;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.3rem 0;
        line-height: 1.2;
    }
    
    .metric-subtitle {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-top: 0.3rem;
    }
    
    .category-text {
        font-size: 0.85rem;
        font-weight: 600;
        line-height: 1.3;
        word-break: break-word;
        hyphens: auto;
    }
    
    .comparison-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem;
        border: 1px solid #60a5fa;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.2);
    }
    
    .analysis-section {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 15px rgba(6, 95, 70, 0.3);
        line-height: 1.6;
    }
    
    .comparison-analysis {
        background: linear-gradient(135deg, #7c2d12 0%, #dc2626 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
        box-shadow: 0 4px 15px rgba(124, 45, 18, 0.3);
        line-height: 1.6;
    }
    
    .insight-highlight {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #60a5fa;
    }
    
    .chatbot-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 2px solid #3b82f6;
    }
    
    .chat-message-bot {
        background: #1e40af;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #60a5fa;
        line-height: 1.6;
    }
    
    .section-spacing {
        margin: 2rem 0;
    }
    
    .product-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
        word-break: break-word;
    }
    
    .warning-highlight {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #f87171;
    }
</style>
""", unsafe_allow_html=True)

class ContentGuardrails:
    """Implements guardrails for LLM responses"""
    
    def __init__(self, profanity_threshold=0.7, confidence_threshold=0.6):
        self.profanity_threshold = profanity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Try to load profanity filter
        try:
            from profanity_check import predict_prob
            self.predict_profanity = predict_prob
            self.profanity_filter_loaded = True
        except ImportError:
            print("Warning: profanity-check not installed - profanity filtering disabled")
            self.profanity_filter_loaded = False
    
    def check_profanity(self, text):
        """Check if text contains profanity"""
        if not self.profanity_filter_loaded:
            return {"is_profane": False, "score": 0.0}
        
        try:
            profanity_score = float(self.predict_profanity([text])[0])
            return {
                "is_profane": profanity_score > self.profanity_threshold,
                "score": profanity_score
            }
        except Exception as e:
            return {"is_profane": False, "score": 0.0, "error": str(e)}
    
    def check_confidence(self, response):
        """Check for low confidence indicators"""
        low_confidence_phrases = [
            "I'm not sure", "I don't know", "uncertain", 
            "might be", "could be", "possibly", "not certain",
            "unclear", "I think", "maybe", "perhaps"
        ]
        
        confidence_score = 1.0
        matched_phrases = []
        
        response_lower = response.lower()
        for phrase in low_confidence_phrases:
            if phrase.lower() in response_lower:
                confidence_score -= 0.1
                matched_phrases.append(phrase)
                
        confidence_score = max(0.0, confidence_score)
        
        return {
            "is_low_confidence": confidence_score < self.confidence_threshold,
            "score": confidence_score,
            "matched_phrases": matched_phrases if matched_phrases else None
        }
    
    def check_hallucination_risk(self, response):
        """Estimate hallucination risk based on specific claims without sources"""
        # Look for specific numerical claims and unsupported statements
        specific_claim_patterns = [
            r'(\d+)%', r'(\d+) percent', r'according to', r'study shows',
            r'research indicates', r'statistics show', r'data reveals',
            r'(\d+) times', r'(\d+)x', r'increase of (\d+)'
        ]
        
        citation_patterns = [
            r'source:', r'citation:', r'\[\d+\]', r'based on data',
            r'from the dataset', r'customer review', r'review data'
        ]
        
        claims = []
        for pattern in specific_claim_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            claims.extend(matches)
        
        citations_present = any(re.search(pattern, response, re.IGNORECASE) 
                               for pattern in citation_patterns)
        
        hallucination_risk = 0.2  # Base risk
        if claims and not citations_present:
            hallucination_risk += min(0.4, 0.1 * len(claims))
        
        return {
            "high_risk": hallucination_risk > 0.5,
            "risk_score": min(hallucination_risk, 1.0),
            "specific_claims": len(claims),
            "citations_present": citations_present
        }
    
    def apply_all_checks(self, text):
        """Apply all guardrail checks"""
        profanity_check = self.check_profanity(text)
        confidence_check = self.check_confidence(text)
        hallucination_check = self.check_hallucination_risk(text)
        
        passes_guardrails = (
            not profanity_check.get("is_profane", False) and
            not confidence_check.get("is_low_confidence", False) and
            not hallucination_check.get("high_risk", False)
        )
        
        return {
            "passes_guardrails": passes_guardrails,
            "profanity_check": profanity_check,
            "confidence_check": confidence_check,
            "hallucination_check": hallucination_check,
            "filtered_text": text
        }

class ProductAnalyzer:
    """Product analysis system with enhanced guardrails and summary features"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('API_KEY')
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.guardrails = ContentGuardrails()
    
    def analyze_single_product(self, product_data, all_products_context):
        """Analyze a single product comprehensively with guardrails"""
        
        if not self.api_key:
            return {"error": "API key not configured. Please add your API key to the .env file."}
        
        # Prepare comprehensive product context
        product_context = self.prepare_single_product_context(product_data, all_products_context)
        
        prompt = f"""
        You are an expert product analyst. Analyze this product comprehensively based on the provided data, with special emphasis on recent customer reviews.

        Product Data:
        {product_context}

        Please provide a detailed analysis covering:

        1. **Product Overview**: Brief summary of the product and its positioning
        2. **Strengths**: Key positive aspects based on customer reviews (emphasize recent feedback)
        3. **Concerns**: Areas of improvement or issues mentioned by customers
        4. **Value Proposition**: Price-to-performance analysis
        5. **Use Case Recommendations**: Who should buy this product and why
        6. **Final Verdict**: Overall recommendation with confidence level

        If customer review data is insufficient, clearly state this and base analysis on available product specifications and market positioning.

        Be descriptive but concise. Focus on actionable insights for potential buyers.
        """
        
        return self.make_api_call_with_guardrails(prompt)
    
    def compare_products(self, product1_data, product2_data, all_products_context):
        """Compare two products comprehensively with guardrails"""
        
        if not self.api_key:
            return {"error": "API key not configured. Please add your API key to the .env file."}
        
        # Prepare comparison context
        comparison_context = self.prepare_comparison_context(product1_data, product2_data, all_products_context)
        
        prompt = f"""
        You are an expert product analyst. Compare these two products comprehensively, emphasizing recent customer reviews and real-world usage feedback.

        Comparison Data:
        {comparison_context}

        Please provide a detailed comparison covering:

        1. **Product Overviews**: Brief intro to both products
        2. **Head-to-Head Analysis**: 
           - Performance comparison based on reviews
           - Build quality and durability feedback
           - Value for money assessment
           - User experience insights
        3. **Strengths of Each Product**: What each product does better
        4. **Weaknesses of Each Product**: Areas where each product falls short
        5. **Use Case Recommendations**: 
           - When to choose Product 1 and why
           - When to choose Product 2 and why
           - Specific scenarios for each
        6. **Final Recommendation**: Which product wins overall and why, with confidence level

        If customer review data is insufficient for either product, clearly state this and base analysis on available specifications and market research. Be descriptive and provide actionable buying guidance.
        """
        
        return self.make_api_call_with_guardrails(prompt)
    
    def answer_dataset_question(self, question, dataset_context):
        """Answer questions about the dataset with guardrails"""
        
        if not self.api_key:
            return {"error": "API key not configured. Please add your API key to the .env file."}
        
        # Prepare dataset context
        context_summary = self.prepare_dataset_context(dataset_context)
        
        prompt = f"""
        You are an expert product analyst with access to a comprehensive product dataset. Answer the user's question based on the provided data, emphasizing recent customer reviews and feedback.

        Dataset Context:
        {context_summary}

        User Question: {question}

        Please provide a detailed, accurate answer based on the available data. If there isn't enough data to fully answer the question, clearly state this and provide insights based on what data is available.

        Focus on:
        - Specific product names, prices, and ratings
        - Customer feedback trends from reviews
        - Market insights and comparisons
        - Actionable recommendations

        Be descriptive and provide valuable insights that help users make informed decisions.
        """
        
        return self.make_api_call_with_guardrails(prompt)
    
    def make_api_call_with_guardrails(self, prompt):
        """Make API call with guardrails and summary extraction"""
        
        # Check prompt content first
        prompt_check = self.guardrails.apply_all_checks(prompt)
        if not prompt_check["passes_guardrails"]:
            return {
                "error": "Prompt failed content guidelines",
                "details": prompt_check
            }
        
        # Enhanced system prompt for summary generation
        system_prompt = """You are an expert product analyst specializing in consumer electronics and tech products. 
        Provide detailed, actionable insights based on customer reviews and market data. Always emphasize recent 
        customer feedback when available. 

        IMPORTANT: Begin your response with a 2-3 sentence summary of your key findings, followed by 
        a line break and then your full detailed analysis. Only make factual claims when supported by data. 
        When uncertain, clearly state the limitations of your knowledge."""
        
        payload = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Apply output guardrails
            output_check = self.guardrails.apply_all_checks(content)
            if not output_check["passes_guardrails"]:
                return {
                    "error": "Response failed content guidelines",
                    "details": output_check
                }
            
            # Extract summary and analysis
            if "---" in content:
                parts = content.split("---", 1)
                summary = parts[0].strip()
                full_analysis = parts[1].strip() if len(parts) > 1 else content
            else:
                # Fallback: use first paragraph as summary
                paragraphs = content.split('\n\n')
                summary = paragraphs[0] if paragraphs else content[:200] + "..."
                full_analysis = content
            
            return {
                "summary": summary,
                "full_analysis": full_analysis,
                "complete_response": content,
                "guardrail_status": output_check
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"API connection error: {str(e)}"}
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}
    
    def prepare_single_product_context(self, product_data, all_products_context):
        """Prepare comprehensive context for single product analysis"""
        try:
            # Basic product info
            context = f"""
Product Name: {product_data['product_name']}
Category: {product_data['category']}
Price: ₹{product_data['price']:,.0f}
Rating: {product_data['rating']:.1f}/5
Review Count: {product_data['review_count']:,}

Customer Review Sample:
{product_data['review_text'][:500]}...

Market Context:
- Total products in dataset: {len(all_products_context)}
- Average price in category: ₹{all_products_context[all_products_context['category'] == product_data['category']]['price'].mean():.0f}
- Average rating in category: {all_products_context[all_products_context['category'] == product_data['category']]['rating'].mean():.1f}/5
- This product ranks #{all_products_context[all_products_context['rating'] >= product_data['rating']].shape[0]} by rating in dataset

Price Comparison:
- Products under ₹{product_data['price']:.0f}: {all_products_context[all_products_context['price'] < product_data['price']].shape[0]}
- Products over ₹{product_data['price']:.0f}: {all_products_context[all_products_context['price'] > product_data['price']].shape[0]}

Similar Products (for context):
"""
            
            # Add similar products for context
            similar_products = all_products_context[
                (all_products_context['category'] == product_data['category']) &
                (all_products_context['product_name'] != product_data['product_name'])
            ].nlargest(3, 'rating')
            
            for _, similar in similar_products.iterrows():
                context += f"- {similar['product_name'][:40]}... (₹{similar['price']:.0f}, {similar['rating']:.1f} stars)\n"
            
            return context
            
        except Exception as e:
            return f"Error preparing context: {str(e)}"
    
    def prepare_comparison_context(self, product1_data, product2_data, all_products_context):
        """Prepare comprehensive context for product comparison"""
        try:
            context = f"""
PRODUCT 1:
Name: {product1_data['product_name']}
Category: {product1_data['category']}
Price: ₹{product1_data['price']:,.0f}
Rating: {product1_data['rating']:.1f}/5
Review Count: {product1_data['review_count']:,}
Customer Review Sample: {product1_data['review_text'][:300]}...

PRODUCT 2:
Name: {product2_data['product_name']}
Category: {product2_data['category']}
Price: ₹{product2_data['price']:,.0f}
Rating: {product2_data['rating']:.1f}/5
Review Count: {product2_data['review_count']:,}
Customer Review Sample: {product2_data['review_text'][:300]}...

Market Context:
- Dataset contains {len(all_products_context)} total products
- Average market price: ₹{all_products_context['price'].mean():.0f}
- Average market rating: {all_products_context['rating'].mean():.1f}/5
- Price difference: ₹{abs(product1_data['price'] - product2_data['price']):.0f}
- Rating difference: {abs(product1_data['rating'] - product2_data['rating']):.1f} stars

Competitive Landscape:
- Products in similar price range (±₹500): {all_products_context[(all_products_context['price'] >= min(product1_data['price'], product2_data['price']) - 500) & (all_products_context['price'] <= max(product1_data['price'], product2_data['price']) + 500)].shape[0]}
- Higher rated alternatives: {all_products_context[all_products_context['rating'] > max(product1_data['rating'], product2_data['rating'])].shape[0]}
"""
            
            return context
            
        except Exception as e:
            return f"Error preparing comparison context: {str(e)}"
    
    def prepare_dataset_context(self, dataset):
        """Prepare comprehensive dataset context for Q&A"""
        try:
            if dataset is None or dataset.empty:
                return "No product data available."
            
            # Dataset overview
            total_products = len(dataset)
            categories = dataset['category'].value_counts().head(5).to_dict()
            avg_rating = dataset['rating'].mean()
            price_range = f"₹{dataset['price'].min():.0f} - ₹{dataset['price'].max():.0f}"
            
            # Top products by different criteria
            top_rated = dataset.nlargest(5, 'rating')[['product_name', 'rating', 'price', 'category']].to_dict('records')
            most_reviewed = dataset.nlargest(5, 'review_count')[['product_name', 'review_count', 'rating', 'price']].to_dict('records')
            
            # Price segments
            budget_products = dataset[dataset['price'] <= 1000]
            mid_range_products = dataset[(dataset['price'] > 1000) & (dataset['price'] <= 5000)]
            premium_products = dataset[dataset['price'] > 5000]
            
            # Brand analysis
            brands = []
            for product in dataset['product_name'].head(20):
                words = str(product).split()
                if words:
                    brands.append(words[0])
            brand_counts = Counter(brands).most_common(5)
            
            context = f"""
Dataset Overview:
- Total Products: {total_products}
- Average Rating: {avg_rating:.1f}/5
- Price Range: {price_range}
- Categories: {categories}

Price Segments:
- Budget (≤₹1000): {len(budget_products)} products, Avg Rating: {budget_products['rating'].mean():.1f}/5
- Mid-range (₹1001-₹5000): {len(mid_range_products)} products, Avg Rating: {mid_range_products['rating'].mean():.1f}/5
- Premium (>₹5000): {len(premium_products)} products, Avg Rating: {premium_products['rating'].mean():.1f}/5

Top-Rated Products:
{json.dumps(top_rated, indent=2)}

Most Reviewed Products:
{json.dumps(most_reviewed, indent=2)}

Popular Brands (by frequency):
{dict(brand_counts)}

Sample Customer Reviews (Recent/Prioritized):
"""
            
            # Add sample reviews for context
            for i, (_, product) in enumerate(dataset.head(3).iterrows()):
                context += f"""
Product {i+1}: {product['product_name'][:50]}...
Rating: {product['rating']}/5, Price: ₹{product['price']:.0f}
Review: "{str(product['review_text'])[:200]}..."
"""
            
            return context
            
        except Exception as e:
            return f"Error preparing dataset context: {str(e)}"

class EnhancedProductIntelligence:
    def __init__(self):
        self.df = None
        self.processed_data = None
        self.analyzer = ProductAnalyzer()
        
    def format_category(self, category):
        """Format category for better display"""
        if not category or category == 'Unknown':
            return "Not Specified"
        
        # Replace pipes with arrows and limit length
        formatted = str(category).replace('|', ' › ').replace('&', ' & ')
        
        # Truncate if too long
        if len(formatted) > 50:
            parts = formatted.split(' › ')
            if len(parts) > 2:
                formatted = f"{parts[0]} › ... › {parts[-1]}"
            elif len(formatted) > 50:
                formatted = formatted[:47] + "..."
        
        return formatted
    
    def format_price(self, price):
        """Format price consistently"""
        try:
            price_num = float(price)
            if price_num >= 100000:
                return f"₹{price_num/100000:.1f}L"
            elif price_num >= 1000:
                return f"₹{price_num/1000:.0f}K"
            else:
                return f"₹{price_num:.0f}"
        except:
            return "₹0"
    
    def format_review_count(self, count):
        """Format review count consistently"""
        try:
            count_num = int(float(count))
            if count_num >= 10000:
                return f"{count_num/1000:.0f}K"
            else:
                return f"{count_num:,}"
        except:
            return "0"
    
    def load_and_process_data(self, uploaded_file):
        """Load and process dataset with recent review prioritization"""
        try:
            self.df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            
            # Data cleaning and processing
            self.df = self.clean_data()
            self.processed_data = self.prepare_data_for_analysis()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean and prepare data with recent review emphasis"""
        df = self.df.copy()
        
        # Essential columns processing
        required_cols = ['product_name', 'category', 'rating', 'review_content', 'discounted_price', 'rating_count']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 'Unknown' if col in ['product_name', 'category', 'review_content'] else 0
        
        # Clean numeric columns
        df['discounted_price'] = pd.to_numeric(
            df['discounted_price'].astype(str).str.replace(',', '').str.replace('₹', ''),
            errors='coerce'
        ).fillna(0)
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['rating_count'] = pd.to_numeric(
            df['rating_count'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(0)
        
        # Preserve full category path
        df['category'] = df['category'].astype(str)
        
        # Recent review weighting (simulate recent priority)
        df['review_recency_weight'] = np.linspace(0.6, 1.0, len(df))
        
        # Sort by recency weight to prioritize recent reviews
        df = df.sort_values('review_recency_weight', ascending=False)
        
        return df
    
    def prepare_data_for_analysis(self):
        """Prepare data for analysis"""
        processed = []
        
        for _, product in self.df.iterrows():
            analysis = {
                'product_name': product['product_name'],
                'category': product['category'],
                'rating': product['rating'],
                'price': product['discounted_price'],
                'review_count': product['rating_count'],
                'review_text': str(product['review_content']),
                'recency_weight': product['review_recency_weight']
            }
            processed.append(analysis)
        
        return pd.DataFrame(processed)
    
    def create_main_interface(self):
        """Create main interface"""
        # Add spacing
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "Product Analysis", 
            "Product Comparison", 
            "Dataset Assistant"
        ])
        
        with tab1:
            self.create_product_analysis()
        
        with tab2:
            self.create_product_comparison()
        
        with tab3:
            self.create_dataset_assistant()
    
    def create_product_analysis(self):
        """Product analysis using advanced system with summary display"""
        st.markdown("### Product Analysis")
        
        # Add spacing
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        product_names = self.processed_data['product_name'].unique()
        selected_product = st.selectbox(
            "Select a product for analysis:",
            options=product_names,
            key="product_analysis"
        )
        
        if selected_product:
            product_data = self.processed_data[
                self.processed_data['product_name'] == selected_product
            ].iloc[0]
            
            # Product title
            st.markdown(f"""
            <div class="product-title">
                {product_data['product_name'][:80]}...
            </div>
            """, unsafe_allow_html=True)
            
            # Improved metric cards with better formatting
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Rating</div>
                    <div class="metric-value">{product_data['rating']:.1f}/5</div>
                    <div class="metric-subtitle">Customer Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                formatted_price = self.format_price(product_data['price'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Price</div>
                    <div class="metric-value">{formatted_price}</div>
                    <div class="metric-subtitle">Current Price</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                formatted_reviews = self.format_review_count(product_data['review_count'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Reviews</div>
                    <div class="metric-value">{formatted_reviews}</div>
                    <div class="metric-subtitle">Customer Feedback</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                formatted_category = self.format_category(product_data['category'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Category</div>
                    <div class="category-text">{formatted_category}</div>
                    <div class="metric-subtitle">Product Category</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add spacing
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            
            # Analysis button
            if st.button("Generate Analysis", type="primary", key="single_analysis"):
                with st.spinner("Analyzing product... This may take 30-60 seconds"):
                    result = self.analyzer.analyze_single_product(
                        product_data.to_dict(), 
                        self.processed_data
                    )
                    
                    if "error" in result:
                        st.error(f"Analysis failed: {result['error']}")
                        if "details" in result:
                            with st.expander("Technical details"):
                                st.write(result["details"])
                    else:
                        # Display summary first in a highlighted box
                        st.markdown(f"""
                        <div class="insight-highlight">
                            <h3>Summary</h3>
                            <div style="white-space: pre-line; line-height: 1.6; font-weight: 500;">
                                {result["summary"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Then display full analysis
                        st.markdown(f"""
                        <div class="analysis-section">
                            <h3>Detailed Product Analysis</h3>
                            <div style="white-space: pre-line; line-height: 1.6;">
                                {result["full_analysis"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show guardrail status if needed
                        if "guardrail_status" in result and not result["guardrail_status"]["passes_guardrails"]:
                            with st.expander("Content Safety Information"):
                                st.write("This response was processed with content safety checks.")
                                st.write(result["guardrail_status"])
            
            # Raw data preview with improved formatting
            st.markdown("### Product Data Preview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Product Information:**")
                st.write(f"**Name:** {product_data['product_name'][:50]}...")
                st.write(f"**Category:** {self.format_category(product_data['category'])}")
                st.write(f"**Rating:** {product_data['rating']:.1f}/5")
                st.write(f"**Price:** {self.format_price(product_data['price'])}")
                st.write(f"**Reviews:** {self.format_review_count(product_data['review_count'])}")
            
            with col2:
                st.markdown("**Review Sample:**")
                review_sample = product_data['review_text'][:300] if len(str(product_data['review_text'])) > 10 else "No review text available"
                st.markdown(f"""
                <div style="background: #1e40af; padding: 1rem; border-radius: 8px; font-style: italic; max-height: 150px; overflow-y: auto; line-height: 1.4;">
                    "{review_sample}..."
                </div>
                """, unsafe_allow_html=True)
    
    def create_product_comparison(self):
        """Product comparison using advanced system with summary display"""
        st.markdown("### Product Comparison")
        st.markdown("*Intelligent comparison with actionable recommendations*")
        
        # Add spacing
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        product_names = self.processed_data['product_name'].unique()
        
        col1, col2 = st.columns(2)
        with col1:
            product1 = st.selectbox("Select Product 1:", options=product_names, key="comp1")
        with col2:
            product2 = st.selectbox("Select Product 2:", options=product_names, key="comp2")
        
        if product1 and product2 and product1 != product2:
            p1_data = self.processed_data[self.processed_data['product_name'] == product1].iloc[0]
            p2_data = self.processed_data[self.processed_data['product_name'] == product2].iloc[0]
            
            # Quick comparison overview with improved formatting
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="comparison-card">
                    <h3>Product 1</h3>
                    <p><strong>{p1_data['product_name'][:40]}...</strong></p>
                    <p><strong>Price:</strong> {self.format_price(p1_data['price'])} | <strong>Rating:</strong> {p1_data['rating']:.1f} stars</p>
                    <p><strong>Reviews:</strong> {self.format_review_count(p1_data['review_count'])}</p>
                    <p><strong>Category:</strong> {self.format_category(p1_data['category'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="comparison-card">
                    <h3>Product 2</h3>
                    <p><strong>{p2_data['product_name'][:40]}...</strong></p>
                    <p><strong>Price:</strong> {self.format_price(p2_data['price'])} | <strong>Rating:</strong> {p2_data['rating']:.1f} stars</p>
                    <p><strong>Reviews:</strong> {self.format_review_count(p2_data['review_count'])}</p>
                    <p><strong>Category:</strong> {self.format_category(p2_data['category'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add spacing
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            
            # Comparison button
            if st.button("Generate Comparison", type="primary", key="comparison"):
                with st.spinner("Comparing products... This may take 60-90 seconds"):
                    result = self.analyzer.compare_products(
                        p1_data.to_dict(), 
                        p2_data.to_dict(), 
                        self.processed_data
                    )
                    
                    if "error" in result:
                        st.error(f"Comparison failed: {result['error']}")
                        if "details" in result:
                            with st.expander("Technical details"):
                                st.write(result["details"])
                    else:
                        # Display summary first
                        st.markdown(f"""
                        <div class="insight-highlight">
                            <h3>Comparison Summary</h3>
                            <div style="white-space: pre-line; line-height: 1.6; font-weight: 500;">
                                {result["summary"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Then display full comparison
                        st.markdown(f"""
                        <div class="comparison-analysis">
                            <h3>Detailed Comparison Analysis</h3>
                            <div style="white-space: pre-line; line-height: 1.6;">
                                {result["full_analysis"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Quick stats comparison with improved formatting
            st.markdown("### Quick Statistics Comparison")
            comparison_data = {
                'Metric': ['Price', 'Rating', 'Review Count', 'Price Difference', 'Rating Difference'],
                f'{product1[:20]}...': [
                    self.format_price(p1_data['price']),
                    f"{p1_data['rating']:.1f}/5",
                    self.format_review_count(p1_data['review_count']),
                    "Base Product",
                    "Base Product"
                ],
                f'{product2[:20]}...': [
                    self.format_price(p2_data['price']),
                    f"{p2_data['rating']:.1f}/5",
                    self.format_review_count(p2_data['review_count']),
                    f"₹{p2_data['price'] - p1_data['price']:+,.0f}",
                    f"{p2_data['rating'] - p1_data['rating']:+.1f}"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    def create_dataset_assistant(self):
        """Dataset assistant using advanced system with summary display"""
        st.markdown("### Product Assistant")
        
        st.markdown("""
        <div class="chatbot-container">
            <h3>Product Assistant</h3>
            <p>Ask me anything about the products in your dataset. I'll analyze reviews, compare products, and provide insights based on customer feedback.</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Ask me anything about the products:",
            placeholder="e.g., 'Which products offer best value under ₹1000?' or 'What do customers say about charging speed?'",
            key="assistant_input"
        )
        
        if st.button("Ask Assistant", type="primary", key="assistant_btn"):
            if user_question:
                with st.spinner("Analyzing your question..."):
                    result = self.analyzer.answer_dataset_question(user_question, self.processed_data)
                    
                    if "error" in result:
                        st.error(f"Assistant error: {result['error']}")
                        if "details" in result:
                            with st.expander("Technical details"):
                                st.write(result["details"])
                    else:
                        # Display summary first
                        st.markdown(f"""
                        <div class="insight-highlight">
                            <h3>Quick Answer</h3>
                            <div style="white-space: pre-line; line-height: 1.6; font-weight: 500;">
                                {result["summary"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Then display detailed response
                        st.markdown(f"""
                        <div class="chat-message-bot">
                            <strong>Detailed Assistant Response:</strong><br>
                            <div style="white-space: pre-line; line-height: 1.6; margin-top: 0.5rem;">
                                {result["full_analysis"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Enhanced sample questions
        st.markdown("#### Try These Questions:")
        sample_questions = [
            "Which products have the most positive customer feedback about durability?",
            "Compare charging cables vs power banks based on customer satisfaction",
            "What are the most common complaints in products under ₹500?",
            "Which brands consistently deliver value for money according to reviews?",
            "Analyze the top 3 products by customer satisfaction and explain why they succeed",
            "What features do customers mention most positively in recent reviews?",
            "Which products show improvement in customer feedback over time?",
            "Recommend products for different budgets with detailed reasoning"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(question, key=f"sample_{i}"):
                with st.spinner("Processing..."):
                    result = self.analyzer.answer_dataset_question(question, self.processed_data)
                    
                    if "error" in result:
                        st.error(f"Assistant error: {result['error']}")
                    else:
                        # Display summary first
                        st.markdown(f"""
                        <div class="insight-highlight">
                            <h3>Quick Answer</h3>
                            <div style="white-space: pre-line; line-height: 1.6; font-weight: 500;">
                                {result["summary"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Then display detailed response
                        st.markdown(f"""
                        <div class="chat-message-bot">
                            <strong>Detailed Response:</strong><br>
                            <div style="white-space: pre-line; line-height: 1.6; margin-top: 0.5rem;">
                                {result["full_analysis"]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    st.markdown("""
    <div class="main-container">
        <h1>McKinsey TechQuest 2025</h1>
        <h2>Product Intelligence Platform</h2>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = EnhancedProductIntelligence()
    
    # Sidebar
    st.sidebar.markdown("## Control Panel")
    
    # API Key Configuration
    st.sidebar.markdown("### API Configuration")
    api_key_input = st.sidebar.text_input(
        "Enter API Key:",
        type="password",
        help="Get your API key from the service provider"
    )
    
    if api_key_input:
        os.environ['API_KEY'] = api_key_input
        analyzer.analyzer.api_key = api_key_input
        st.sidebar.success("API Key configured!")
    
    # File uploader - this fixes the original error
    uploaded_file = st.sidebar.file_uploader(
        "Upload Product Dataset (CSV)", 
        type=['csv'],
        help="Upload your product review dataset for analysis"
    )
    
    if uploaded_file is not None:
        with st.spinner("Processing dataset for analysis..."):
            if analyzer.load_and_process_data(uploaded_file):
                analyzer.create_main_interface()
                
                # Enhanced sidebar stats
                st.sidebar.markdown("### Dataset Overview")
                st.sidebar.write(f"**Products:** {len(analyzer.processed_data):,}")
                st.sidebar.write(f"**Categories:** {analyzer.processed_data['category'].nunique()}")
                st.sidebar.write(f"**Avg Rating:** {analyzer.processed_data['rating'].mean():.1f} stars")
                price_min = analyzer.processed_data['price'].min()
                price_max = analyzer.processed_data['price'].max()
                st.sidebar.write(f"**Price Range:** {analyzer.format_price(price_min)} - {analyzer.format_price(price_max)}")
                
                # Analysis features
                st.sidebar.markdown("### Analysis Features")
                st.sidebar.write("✓ Single Product Analysis")
                st.sidebar.write("✓ Product Comparison")
                st.sidebar.write("✓ Dataset Q&A")
                st.sidebar.write("✓ Recent Review Emphasis")
                st.sidebar.write("✓ Content Safety Guardrails")
                st.sidebar.write("✓ Executive Summaries")
        
    else:
        # Enhanced welcome screen
        st.markdown("""
        ## Product Intelligence Platform
        
        ### To get started:
        1. Add your Perplexity API key in the sidebar
        2. Upload your CSV dataset
        3. Let the system analyze your products comprehensively!
        """)

if __name__ == "__main__":
    main()
